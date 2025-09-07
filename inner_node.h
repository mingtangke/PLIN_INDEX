//
// Copyright (c) Zhou Zhang.
// Implementation of PLIN inner nodes
//

#pragma once
#include "leaf_node.h"
#include "cache_model.h"
class InnerNode{
#pragma pack(push)
#pragma pack(1)

    // The size of header should less than NODE_HEADER_SIZE (256B)
    struct InnerNodeMetadata {
        double slope = 0;
        float intercept = 0;
        uint32_t block_number = 0;
        uint64_t record_number = 0;
        _key_t first_key = FREE_FLAG;
        uint32_t level = 0;
    } inner_node;
    char unused[NODE_HEADER_SIZE - sizeof(InnerNodeMetadata)];

#pragma pack(pop)

    // InnerSlot inner_slots[1]; //utils.h
    InnerSlot *inner_slots;

public:

    constexpr static uint64_t InnerSlotsPerBlock = BLOCK_SIZE / (sizeof(InnerSlot));

    inline uint32_t predict_block (_key_t key, double slope, float intercept, uint32_t block_number) const {
        int64_t predicted_block = (key * slope + intercept + 0.5) / InnerSlotsPerBlock;
        if (predicted_block < 0)
            return 0;
        else if (predicted_block > block_number - 4)
            return block_number - 2;
        return predicted_block + 1;
    }

    InnerNode(InnerSlot& accelerator, uint64_t block_number, _key_t* keys, InnerSlot* nodes, uint64_t number, uint64_t start_pos, double slope, double intercept, uint32_t level, bool rebuild = false) {
        // assert((uint64_t)inner_slots - (uint64_t)&inner_node == NODE_HEADER_SIZE);
        inner_node.block_number = block_number;
        inner_node.slope = slope / INNER_NODE_INIT_RATIO;
        inner_node.intercept = (intercept - start_pos - keys[start_pos] * slope) / INNER_NODE_INIT_RATIO;
        inner_node.record_number = number;
        inner_node.first_key = keys[start_pos];
        inner_node.level = level;
        do_flush(&inner_node, sizeof(inner_node));
        model_correction(inner_node.slope, inner_node.intercept, (inner_node.block_number - 3) * InnerSlotsPerBlock, keys[start_pos], keys[start_pos + number - 1]);
        // p.intercept = inner_node.intercept;
        // p.slope = inner_node.slope;
        // p.first_key = inner_node.first_key;
        
        inner_slots = new InnerSlot[8*block_number];
        // Init
        for (uint64_t i = 0; i < inner_node.block_number; ++i) {
            for (uint8_t j = 0; j < 8; j++) {
                inner_slots[i * InnerSlotsPerBlock + j].min_key = FREE_FLAG;
            }
        }
        // Model-based data placement
        for (uint64_t i = 0; i < number; ++i) {
            data_placement(keys[start_pos + i], nodes[start_pos + i]);
        }
        do_flush(inner_slots, block_number * BLOCK_SIZE);
        // Build accelerator
        accelerator.min_key = inner_node.first_key;
        // accelerator.ptr = galc->relative(this);
        accelerator.ptr = this;
        accelerator.slope = inner_node.slope;
        accelerator.intercept = inner_node.intercept;
        accelerator.set_block_number(inner_node.block_number);
        accelerator.set_type(1);
        accelerator.init_lock();
    }
    
    ~InnerNode() {}


    void destroy () {
        if (inner_node.level > 1) {
            uint64_t slot_number = inner_node.block_number * InnerSlotsPerBlock;
            for (uint64_t slot = 0; slot < slot_number; ++slot) {
                if (inner_slots[slot].min_key != FREE_FLAG) {
                    // reinterpret_cast<InnerNode*>(galc->absolute(inner_slots[slot].ptr))->destroy();
                    reinterpret_cast<InnerNode*>(inner_slots[slot].ptr)->destroy();
                }
            }
        }
        // free(this);bug
        delete this;
    }

    InnerSlot * find_leaf_node (_key_t key, const InnerSlot * accelerator) {
        uint32_t block_number = accelerator->block_number();
        uint64_t slot = predict_block(key, accelerator->slope, accelerator->intercept, block_number) * InnerSlotsPerBlock;
        // Search left
        if (inner_slots[slot].min_key == FREE_FLAG || key < inner_slots[slot].min_key) {
            while (inner_slots[--slot].min_key == FREE_FLAG) {}
        }
        // Search right
        else {
            uint32_t slot_number = block_number * InnerSlotsPerBlock;
            while (slot < slot_number && inner_slots[slot].min_key != FREE_FLAG && key >= inner_slots[slot].min_key) {
                ++slot;
            }
            --slot;
        }
        if (inner_slots[slot].type()) {
            return reinterpret_cast<InnerNode*>(inner_slots[slot].ptr)->find_leaf_node(key, &inner_slots[slot]);
        }
        else {
            return &inner_slots[slot];
        }
    }

    // Upsert new nodes,return 1:update , 2:insert ,3:failure
    uint32_t upsert_node (InnerSlot& node, InnerSlot * accelerator) {
        uint32_t block_number = accelerator->block_number();
        uint64_t slot = predict_block(node.min_key, accelerator->slope, accelerator->intercept, block_number) * InnerSlotsPerBlock;
        uint64_t predicted_slot = slot;
        // Search left
        if (inner_slots[slot].min_key == FREE_FLAG || node.min_key < inner_slots[slot].min_key) {
            while (inner_slots[--slot].min_key == FREE_FLAG) {}
        }
        // Search right
        else {
            uint32_t slot_number = block_number * InnerSlotsPerBlock;
            while (slot < slot_number && inner_slots[slot].min_key != FREE_FLAG && node.min_key >= inner_slots[slot].min_key) {
                ++slot;
            }
            --slot;
        }

        if (inner_slots[slot].type()) {//inner node
            return reinterpret_cast<InnerNode*>(inner_slots[slot].ptr)->upsert_node(node, &inner_slots[slot]);
        }
        // Upsert node
        else {
            if (inner_slots[slot].min_key == node.min_key) {//the range of the node is the same，upgate the node
                accelerator->get_lock();
                inner_slots[slot] = node;//leaf
                do_flush(&inner_slots[slot], sizeof(InnerSlot));
                mfence();
                accelerator->release_lock();
                return 1;
            }
            //have enough space
            //如果该位置为空，说明在这个区域内有足够的空间来插入新的节点，因为是有序的，所以至少可以插入一个值
            else if (inner_slots[predicted_slot + InnerSlotsPerBlock - 1].min_key == FREE_FLAG) {
                accelerator->get_lock();
                uint64_t target_slot = predicted_slot;
                if (slot >= predicted_slot) {
                    while (inner_slots[++target_slot].min_key <= node.min_key) {}//break point
                }
                slot = predicted_slot + InnerSlotsPerBlock - 1;
                while (inner_slots[--slot].min_key == FREE_FLAG) {}
                for (++slot; slot > target_slot; --slot) {
                    inner_slots[slot] = inner_slots[slot - 1];//move one by one
                }
                inner_slots[target_slot] = node;//insert
                do_flush(&inner_slots[predicted_slot], InnerSlotsPerBlock * sizeof(InnerSlot));
                mfence();
                accelerator->release_lock();
                return 2;//insert succeed
            }
            else {
                return 3;//fail to upsert
            }
        }
    }

    void data_placement (_key_t key, InnerSlot node) {
        uint64_t slot = predict_block(key, inner_node.slope, inner_node.intercept, inner_node.block_number) * InnerSlotsPerBlock;
        for (uint32_t i = 0; i < InnerSlotsPerBlock; ++i) {
            if (inner_slots[slot + i].min_key == FREE_FLAG) {                
                inner_slots[slot + i] = node;
                return;
            }
        }
        //not in the predicted block,search in the following all the slots of all the blocks
        for (slot += InnerSlotsPerBlock; slot < inner_node.block_number * InnerSlotsPerBlock; slot++) {
            if (inner_slots[slot].min_key == FREE_FLAG) {                
                inner_slots[slot] = node;
                return;
            }
        }
        assert(slot < inner_node.block_number * InnerSlotsPerBlock);
        return;
    }

    LeafNode* get_leftmost_leaf(){
        uint64_t slot = 0;
        while(inner_slots[slot].min_key == FREE_FLAG) {
            ++slot;
        }
        if (inner_slots[slot].type() == 0) {
            return reinterpret_cast<LeafNode*>(inner_slots[slot].ptr);
        }
        else {
            return reinterpret_cast<InnerNode*>(inner_slots[slot].ptr)->get_leftmost_leaf();
        }
    }

};

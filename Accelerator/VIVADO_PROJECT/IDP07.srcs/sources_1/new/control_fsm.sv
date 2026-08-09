/*
`timescale 1ns/1ps

import parameter_pkg::*;
//| Improvement                   | Reason                                      |
//| ----------------------------- | ------------------------------------------- |
//| separate START states         | guarantees clean 1-cycle layer start pulse  |
//| done_seen flags               | prevents double-triggering from sticky done |
//| status_error                  | catches illegal state corruption            |
//| protected start latch         | ignores duplicate starts while busy         |
//| explicit output request state | cleaner AXI timing                          |
//| reset-safe sequencing         | avoids partial pipeline restart             |
//| clean pulse clearing          | avoids accidental multi-cycle strobes       |

module control_fsm
    
(

    input logic clk,
    input logic rst_n,

    // =========================================================
    // Host Control
    // =========================================================

    input logic ctrl_start,
    input logic ctrl_reset,

    // =========================================================
    // Feature Buffer
    // =========================================================

    input  logic feat_full,
    output logic feat_clear,

    // =========================================================
    // Layer Control
    // =========================================================

    output logic l1_start,
    input  logic l1_done,

    output logic l2_start,
    input  logic l2_done,

    output logic l3_start,
    input  logic l3_done,

    // =========================================================
    // AXI Output
    // =========================================================

    output logic send_req,
    input  logic output_busy,

    output logic logit_we,

    // =========================================================
    // Final Logits
    // =========================================================

    input logic signed [ACC_W-1:0]
        l3_logits [L3_OUT-1:0],

    output logic signed [ACC_W-1:0]
        out_logits [L3_OUT-1:0],

    // =========================================================
    // Status
    // =========================================================

    output logic status_idle,
    output logic status_busy,
    output logic status_done,
    output logic status_error,

    // =========================================================
    // Metrics
    // =========================================================

    output logic [31:0] inf_counter,
    output logic [31:0] latency_cycles

);

    // =========================================================================
    // FSM
    // =========================================================================

    typedef enum logic [3:0] {

        S_IDLE,
        S_L1_START,
        S_L1_WAIT,
        S_L2_START,
        S_L2_WAIT,
        S_L3_START,
        S_L3_WAIT,
        S_OUTPUT_REQ,
        S_OUTPUT_WAIT,
        S_DONE,
        S_ERROR

    } state_t;

    state_t state;

    // =========================================================================
    // Internal
    // =========================================================================

    logic start_latch;

    logic [31:0] lat_ctr;

    logic l1_done_seen;
    logic l2_done_seen;
    logic l3_done_seen;

    // =========================================================================
    // Sequential FSM
    // =========================================================================

    always_ff @(posedge clk or negedge rst_n) begin

        if(!rst_n) begin

            state <= S_IDLE;

            feat_clear <= 1'b0;

            l1_start <= 1'b0;
            l2_start <= 1'b0;
            l3_start <= 1'b0;

            send_req <= 1'b0;

            logit_we <= 1'b0;

            status_idle  <= 1'b1;
            status_busy  <= 1'b0;
            status_done  <= 1'b0;
            status_error <= 1'b0;

            inf_counter    <= '0;
            latency_cycles <= '0;

            lat_ctr <= '0;

            start_latch <= 1'b0;

            l1_done_seen <= 1'b0;
            l2_done_seen <= 1'b0;
            l3_done_seen <= 1'b0;

            for(int i=0;i<L3_OUT;i++)
                out_logits[i] <= '0;

        end
        else begin

            // =============================================================
            // Default Pulse Clears
            // =============================================================

            feat_clear <= 1'b0;

            l1_start <= 1'b0;
            l2_start <= 1'b0;
            l3_start <= 1'b0;

            send_req <= 1'b0;

            logit_we <= 1'b0;

            status_done <= 1'b0;

            // =============================================================
            // Start Latch
            // =============================================================

            if(ctrl_start && !status_busy)
                start_latch <= 1'b1;

            // =============================================================
            // Soft Reset
            // =============================================================

            if(ctrl_reset) begin

                state <= S_IDLE;

                start_latch <= 1'b0;

                status_idle  <= 1'b1;
                status_busy  <= 1'b0;
                status_done  <= 1'b0;
                status_error <= 1'b0;

                l1_done_seen <= 1'b0;
                l2_done_seen <= 1'b0;
                l3_done_seen <= 1'b0;

            end
            else begin

                case(state)

                    // =====================================================
                    // IDLE
                    // =====================================================

                    S_IDLE: begin

                        status_idle  <= 1'b1;
                        status_busy  <= 1'b0;
                        status_error <= 1'b0;

                        lat_ctr <= '0;

                        l1_done_seen <= 1'b0;
                        l2_done_seen <= 1'b0;
                        l3_done_seen <= 1'b0;

                        if(start_latch && feat_full) begin

                            start_latch <= 1'b0;

                            status_idle <= 1'b0;
                            status_busy <= 1'b1;

                            state <= S_L1_START;

                        end

                    end

                    // =====================================================
                    // L1 START
                    // =====================================================

                    S_L1_START: begin

                        l1_start <= 1'b1;

                        lat_ctr <= lat_ctr + 1'b1;

                        state <= S_L1_WAIT;

                    end

                    // =====================================================
                    // L1 WAIT
                    // =====================================================

                    S_L1_WAIT: begin

                        lat_ctr <= lat_ctr + 1'b1;

                        if(l1_done && !l1_done_seen) begin

                            l1_done_seen <= 1'b1;

                            state <= S_L2_START;

                        end

                    end

                    // =====================================================
                    // L2 START
                    // =====================================================

                    S_L2_START: begin

                        l2_start <= 1'b1;

                        lat_ctr <= lat_ctr + 1'b1;

                        state <= S_L2_WAIT;

                    end

                    // =====================================================
                    // L2 WAIT
                    // =====================================================

                    S_L2_WAIT: begin

                        lat_ctr <= lat_ctr + 1'b1;

                        if(l2_done && !l2_done_seen) begin

                            l2_done_seen <= 1'b1;

                            state <= S_L3_START;

                        end

                    end

                    // =====================================================
                    // L3 START
                    // =====================================================

                    S_L3_START: begin

                        l3_start <= 1'b1;

                        lat_ctr <= lat_ctr + 1'b1;

                        state <= S_L3_WAIT;

                    end

                    // =====================================================
                    // L3 WAIT
                    // =====================================================

                    S_L3_WAIT: begin

                        lat_ctr <= lat_ctr + 1'b1;

                        if(l3_done && !l3_done_seen) begin

                            l3_done_seen <= 1'b1;

                            state <= S_OUTPUT_REQ;

                        end

                    end

                    // =====================================================
                    // OUTPUT REQUEST
                    // =====================================================

                    S_OUTPUT_REQ: begin

                        lat_ctr <= lat_ctr + 1'b1;

                        for(int i=0;i<L3_OUT;i++)
                            out_logits[i] <= l3_logits[i];

                        logit_we <= 1'b1;

                        send_req <= 1'b1;

                        state <= S_OUTPUT_WAIT;

                    end

                    // =====================================================
                    // OUTPUT WAIT
                    // =====================================================

                    S_OUTPUT_WAIT: begin

                        lat_ctr <= lat_ctr + 1'b1;

                        if(output_busy)
                            state <= S_DONE;

                    end

                    // =====================================================
                    // DONE
                    // =====================================================

                    S_DONE: begin

                        latency_cycles <= lat_ctr;

                        inf_counter <=
                            inf_counter + 1'b1;

                        feat_clear <= 1'b1;

                        status_busy <= 1'b0;

                        status_done <= 1'b1;

                        state <= S_IDLE;

                    end

                    // =====================================================
                    // ERROR
                    // =====================================================

                    S_ERROR: begin

                        status_busy  <= 1'b0;
                        status_error <= 1'b1;

                        if(ctrl_reset)
                            state <= S_IDLE;

                    end

                    // =====================================================
                    // DEFAULT
                    // =====================================================

                    default: begin

                        state <= S_ERROR;

                    end

                endcase

            end

        end

    end

endmodule
*/

`timescale 1ns/1ps

module control_fsm
    import parameter_pkg::*;
(

    input logic clk,
    input logic rst_n,

    // =========================================================================
    // HOST CONTROL
    // =========================================================================

    input logic ctrl_start,
    input logic ctrl_reset,

    // =========================================================================
    // FEATURE BUFFER
    // =========================================================================

    input  logic feat_full,
    output logic feat_clear,

    // =========================================================================
    // LAYER CONTROL
    // =========================================================================

    output logic l1_start,
    input  logic l1_done,

    output logic l2_start,
    input  logic l2_done,

    output logic l3_start,
    input  logic l3_done,

    // =========================================================================
    // AXI STREAM MASTER
    // =========================================================================

    output logic send_req,
    input  logic tx_done,

    // =========================================================================
    // LOGIT STORAGE
    // =========================================================================

    output logic logit_we,

    input logic signed [ACC_W-1:0]
        l3_logits [L3_OUT-1:0],

    output logic signed [ACC_W-1:0]
        out_logits [L3_OUT-1:0],

    // =========================================================================
    // STATUS
    // =========================================================================

    output logic status_idle,
    output logic status_busy,
    output logic status_done,
    output logic status_error,

    // =========================================================================
    // METRICS
    // =========================================================================

    output logic [31:0] inf_counter,
    output logic [31:0] latency_cycles

);

    // =========================================================================
    // FSM STATES
    // =========================================================================

    typedef enum logic [3:0] {

        S_IDLE,
        S_L1_START,
        S_L1_WAIT,
        S_L2_START,
        S_L2_WAIT,
        S_L3_START,
        S_L3_WAIT,
        S_OUTPUT_REQ,
        S_OUTPUT_WAIT,
        S_DONE,
        S_ERROR

    } state_t;

    state_t state;

    // =========================================================================
    // INTERNALS
    // =========================================================================

    logic start_latch;

    logic ctrl_start_r;

    logic [31:0] lat_ctr;

    logic [31:0] timeout_ctr;

    localparam int TIMEOUT_CYCLES = 32'd1000000;

    // =========================================================================
    // FSM
    // =========================================================================

    always_ff @(posedge clk or negedge rst_n) begin

        if(!rst_n) begin

            state <= S_IDLE;

            feat_clear <= 1'b0;

            l1_start <= 1'b0;
            l2_start <= 1'b0;
            l3_start <= 1'b0;

            send_req <= 1'b0;

            logit_we <= 1'b0;

            status_idle  <= 1'b1;
            status_busy  <= 1'b0;
            status_done  <= 1'b0;
            status_error <= 1'b0;

            inf_counter    <= '0;
            latency_cycles <= '0;

            lat_ctr <= '0;

            timeout_ctr <= '0;

            start_latch <= 1'b0;

            ctrl_start_r <= 1'b0;

            for(int i=0;i<L3_OUT;i++)
                out_logits[i] <= '0;

        end
        else begin

            // =============================================================
            // DEFAULT PULSES
            // =============================================================

            feat_clear <= 1'b0;

            l1_start <= 1'b0;
            l2_start <= 1'b0;
            l3_start <= 1'b0;

            send_req <= 1'b0;

            logit_we <= 1'b0;

            // =============================================================
            // EDGE DETECT
            // =============================================================

            ctrl_start_r <= ctrl_start;

            // =============================================================
            // START LATCH
            // =============================================================

            if(ctrl_start &&
               !ctrl_start_r &&
               !status_busy) begin

                start_latch <= 1'b1;

                status_done <= 1'b0;

            end

            // =============================================================
            // SOFT RESET
            // =============================================================

            if(ctrl_reset) begin

                state <= S_IDLE;

                feat_clear <= 1'b1;

                start_latch <= 1'b0;

                status_idle  <= 1'b1;
                status_busy  <= 1'b0;
                status_done  <= 1'b0;
                status_error <= 1'b0;

                timeout_ctr <= '0;

                lat_ctr <= '0;

            end
            else begin

                case(state)

                    // =====================================================
                    // IDLE
                    // =====================================================

                    S_IDLE: begin

                        status_idle  <= 1'b1;
                        status_busy  <= 1'b0;
                        status_error <= 1'b0;

                        timeout_ctr <= '0;

                        lat_ctr <= '0;

                        if(start_latch && feat_full) begin

                            start_latch <= 1'b0;

                            status_idle <= 1'b0;
                            status_busy <= 1'b1;

                            timeout_ctr <= '0;

                            state <= S_L1_START;

                        end

                    end

                    // =====================================================
                    // L1 START
                    // =====================================================

                    S_L1_START: begin

                        l1_start <= 1'b1;

                        lat_ctr <= lat_ctr + 1'b1;

                        timeout_ctr <= '0;

                        state <= S_L1_WAIT;

                    end

                    // =====================================================
                    // L1 WAIT
                    // =====================================================

                    S_L1_WAIT: begin

                        lat_ctr <= lat_ctr + 1'b1;

                        timeout_ctr <= timeout_ctr + 1'b1;

                        if(l1_done) begin

                            timeout_ctr <= '0;

                            state <= S_L2_START;

                        end
                        else if(timeout_ctr > TIMEOUT_CYCLES) begin

                            state <= S_ERROR;

                        end

                    end

                    // =====================================================
                    // L2 START
                    // =====================================================

                    S_L2_START: begin

                        l2_start <= 1'b1;

                        lat_ctr <= lat_ctr + 1'b1;

                        timeout_ctr <= '0;

                        state <= S_L2_WAIT;

                    end

                    // =====================================================
                    // L2 WAIT
                    // =====================================================

                    S_L2_WAIT: begin

                        lat_ctr <= lat_ctr + 1'b1;

                        timeout_ctr <= timeout_ctr + 1'b1;

                        if(l2_done) begin

                            timeout_ctr <= '0;

                            state <= S_L3_START;

                        end
                        else if(timeout_ctr > TIMEOUT_CYCLES) begin

                            state <= S_ERROR;

                        end

                    end

                    // =====================================================
                    // L3 START
                    // =====================================================

                    S_L3_START: begin

                        l3_start <= 1'b1;

                        lat_ctr <= lat_ctr + 1'b1;

                        timeout_ctr <= '0;

                        state <= S_L3_WAIT;

                    end

                    // =====================================================
                    // L3 WAIT
                    // =====================================================

                    S_L3_WAIT: begin

                        lat_ctr <= lat_ctr + 1'b1;

                        timeout_ctr <= timeout_ctr + 1'b1;

                        if(l3_done) begin

                            timeout_ctr <= '0;

                            state <= S_OUTPUT_REQ;

                        end
                        else if(timeout_ctr > TIMEOUT_CYCLES) begin

                            state <= S_ERROR;

                        end

                    end

                    // =====================================================
                    // OUTPUT REQUEST
                    // =====================================================

                    S_OUTPUT_REQ: begin

                        lat_ctr <= lat_ctr + 1'b1;

                        timeout_ctr <= '0;

                        for(int i=0;i<L3_OUT;i++)
                            out_logits[i] <= l3_logits[i];

                        logit_we <= 1'b1;

                        send_req <= 1'b1;

                        state <= S_OUTPUT_WAIT;

                    end

                    // =====================================================
                    // OUTPUT WAIT
                    // =====================================================

                    S_OUTPUT_WAIT: begin

                        lat_ctr <= lat_ctr + 1'b1;

                        timeout_ctr <= timeout_ctr + 1'b1;

                        if(tx_done) begin

                            timeout_ctr <= '0;

                            state <= S_DONE;

                        end
                        else if(timeout_ctr > TIMEOUT_CYCLES) begin

                            state <= S_ERROR;

                        end

                    end

                    // =====================================================
                    // DONE
                    // =====================================================

                    S_DONE: begin

                        latency_cycles <= lat_ctr;

                        inf_counter <=
                            inf_counter + 1'b1;

                        feat_clear <= 1'b1;

                        status_busy <= 1'b0;

                        status_done <= 1'b1;

                        timeout_ctr <= '0;

                        state <= S_IDLE;

                    end

                    // =====================================================
                    // ERROR
                    // =====================================================

                    S_ERROR: begin

                        status_busy  <= 1'b0;

                        status_error <= 1'b1;

                        feat_clear <= 1'b1;

                        if(ctrl_reset)
                            state <= S_IDLE;

                    end

                    // =====================================================
                    // DEFAULT RECOVERY
                    // =====================================================

                    default: begin

                        state <= S_ERROR;

                    end

                endcase

            end

        end

    end

endmodule
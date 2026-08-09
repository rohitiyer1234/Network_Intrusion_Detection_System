// =============================================================================
// mac_array.sv
// -----------------------------------------------------------------------------
// Parallel MAC array for MLP inference.
//
// ARCHITECTURE
// -----------------------------------------------------------------------------
// - PARALLELISM = number of neurons computed simultaneously
// - IN_DIM      = number of accumulation steps per neuron
//
// Example:
//   Layer: 20 -> 64
//   PARALLELISM = 32
//
//   Pass 1 computes neurons [0:31]
//   Pass 2 computes neurons [32:63]
//
// Each MAC accumulates sequentially across all IN_DIM inputs.
//
// -----------------------------------------------------------------------------
// TIMING MODEL
// -----------------------------------------------------------------------------
//
// Weight BRAM latency = 1 cycle.
//
// mlp_layer performs weight prefetch:
//
//   cycle N-1 : drive BRAM address for input N
//   cycle N   : wt_data valid for input N
//
// mac_array at cycle N:
//   cur_in_r = in_vec[N]
//   wt_data  = weights for input N across all neurons
//
// mac_unit internal pipeline:
//
//   stage1 : registered multiply
//   stage2 : registered accumulation
//
// Therefore:
//
//   feed cycles  : 0 .. IN_DIM-1
//   drain cycles : 1 additional cycle
//
// Final accumulation completes at:
//   step == IN_DIM
//
// done asserted when:
//   step == STEP_DONE
//
// -----------------------------------------------------------------------------
// IMPORTANT FIXES
// -----------------------------------------------------------------------------
//
// FIXED:
//   - stale MAC operations during drain
//   - incorrect pipeline drain count
//   - clear_acc alignment
//   - timing fanout from combinational in_vec indexing
//   - output capture race
//   - robust feed/drain separation
//   - synthesis-safe indexing
//
// =============================================================================

`timescale 1ns/1ps

import parameter_pkg::*;

module mac_array #(

    parameter int PARALLELISM = parameter_pkg::PARALLELISM,
    parameter int IN_DIM      = L1_IN,
    parameter int IN_WIDTH    = FEAT_W,
    parameter int WT_WIDTH    = W_WIDTH,
    parameter int ACC_WIDTH   = ACC_W,
    parameter int ACTIVE_MACS = PARALLELISM

)(

    input  logic                              clk,
    input  logic                              rst_n,
    input  logic                              start,
    input  logic signed [IN_WIDTH-1:0]        in_vec [IN_DIM-1:0],
    input  logic signed [WT_WIDTH-1:0]        wt_data [PARALLELISM-1:0],
    output logic signed [ACC_WIDTH-1:0]       acc_out [PARALLELISM-1:0],
    output logic                              done
);

    // =========================================================================
    // Parameters
    // =========================================================================

    localparam int MAC_LATENCY = 1;
    localparam int STEP_DONE = IN_DIM + 1;   // number of steps that the unit has to run for (cycles)
    localparam int CTR_W = $clog2(IN_DIM + MAC_LATENCY + 2);  

    localparam int IDX_W = (IN_DIM <= 1) ? 1 : $clog2(IN_DIM);

    // =========================================================================
    // Control
    // =========================================================================

    logic [CTR_W-1:0] step;
    logic running;
    logic feed_active;

    // Only valid during real input feed cycles
    assign feed_active = running && (step < CTR_W'(IN_DIM));

    // =========================================================================
    // Registered Input Broadcast
    // =========================================================================

    logic signed [IN_WIDTH-1:0] cur_in;

    always_comb begin
        cur_in = '0;
        if (running && step < CTR_W'(IN_DIM))
            cur_in = in_vec[step[IDX_W-1:0]];
        else
            cur_in = '0;
    end

    // =========================================================================
    // MAC Control
    // =========================================================================

    logic mac_en;
    logic mac_clear;

    // MAC active only during valid feed cycles
    assign mac_en = feed_active;

    // Clear only on first valid feed cycle
    assign mac_clear = feed_active && (step == CTR_W'(0));
        

    // =========================================================================
    // MAC Outputs
    // =========================================================================

    logic signed [ACC_WIDTH-1:0] mac_acc [PARALLELISM-1:0];
        

    // =========================================================================
    // MAC Array
    // =========================================================================

    for (genvar k = 0; k < PARALLELISM; k++) begin : gen_macs

        mac_unit #(
            .IN_WIDTH (IN_WIDTH),
            .WT_WIDTH (WT_WIDTH),
            .ACC_WIDTH(ACC_WIDTH)
        ) u_mac (

            .clk      (clk),
            .rst_n    (rst_n),
            .en       (mac_en),
            .clear_acc(mac_clear),
            .a        (cur_in),
            .w        (wt_data[k]),
            .acc      (mac_acc[k])
        );

    end

    // =========================================================================
    // Output Mapping
    // =========================================================================

    always_comb begin

        for (int i = 0; i < PARALLELISM; i++) begin

            if (i < ACTIVE_MACS)
                acc_out[i] = mac_acc[i];
            else
                acc_out[i] = '0;
        end
    end

    // FSM
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      step    <= '0;
      running <= 1'b0;
      done    <= 1'b0;
      for (int i = 0; i < PARALLELISM; i++) acc_out[i] <= '0;
    end else begin
      done <= 1'b0;

      if (start && !running) begin
        step    <= '0;
        running <= 1'b1;
      end else if (running) begin
        if (step == CTR_W'(STEP_DONE)) begin
          running <= 1'b0;
          done    <= 1'b1;
          for (int i = 0; i < PARALLELISM; i++) begin
            if (i < ACTIVE_MACS)
              acc_out[i] <= mac_acc[i];
            else
              acc_out[i] <= '0;
          end
        end else begin
          step <= step + 1;
        end
      end
    end
  end
  
endmodule
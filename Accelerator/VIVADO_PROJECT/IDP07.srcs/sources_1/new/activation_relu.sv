`timescale 1ns/1ps

// =============================================================================
// activation_relu_quant.sv
// -----------------------------------------------------------------------------
// Pipelined ReLU + Requantization block for quantized MLP inference.
//
// DESIGN GOALS
// -----------------------------------------------------------------------------
// - FPGA timing-safe
// - Synthesizable on Vivado
// - Multi-cycle pipeline safe
// - High parallelism friendly
// - Matches QAT training pipeline
// - Reduced combinational depth
// - Reduced routing fanout
// - Correct signed arithmetic
// - SHIFT=0 safe
//
// -----------------------------------------------------------------------------
// PIPELINE
// -----------------------------------------------------------------------------
//
// STAGE 1
//   acc + bias
//   ReLU clamp
//   REGISTER
//
// STAGE 2
//   arithmetic shift
//   int8 saturation
//   dout register
//
// Total latency:
//   en ---> valid = 2 cycles
//
// -----------------------------------------------------------------------------
// TRAINING ALIGNMENT
// -----------------------------------------------------------------------------
//
// Matches:
//
//   relu_int16(x):
//       clamp(0,8192)
//
//   requantize(x, shift)
//
// from QAT training pipeline.
//
// =============================================================================

module activation_relu_quant
    import parameter_pkg::*;
#(

    parameter int N         = L1_OUT,
    parameter int IN_WIDTH  = ACC_W,
    parameter int B_WIDTH   = BIAS_W,

    parameter int SHIFT     = 1

)(

    input  logic                         clk,
    input  logic                         rst_n,
    input  logic                         en,

    input  logic signed [IN_WIDTH-1:0] acc [N-1:0],
    input  logic signed [B_WIDTH-1:0] bias [N-1:0],
    
    // requantized INT8 activations
    output logic signed [7:0] dout [N-1:0],
    output logic                        valid
);

    // =========================================================================
    // ReLU Clamp Limit
    // =========================================================================

    localparam logic signed [IN_WIDTH-1:0] RELU_MAX = $signed(8192);
        

    // =========================================================================
    // Stage 1 Registers
    // =========================================================================

    logic signed [IN_WIDTH-1:0] relu_r [N-1:0];
    logic valid_s1;

    // =========================================================================
    // Stage 2 Registers
    // =========================================================================

    logic valid_s2;

    // =========================================================================
    // Stage 1
    // Bias Add + ReLU Clamp
    // =========================================================================

    always_ff @(posedge clk) begin

        if (!rst_n) begin

            valid_s1 <= 1'b0;

            for (int i = 0; i < N; i++) begin
                relu_r[i] <= '0;
            end

        end
        else begin

            valid_s1 <= en;

            if (en) begin

                for (int i = 0; i < N; i++) begin

                    logic signed [IN_WIDTH-1:0] pre_val;

                    // ---------------------------------------------------------
                    // Bias addition
                    // ---------------------------------------------------------

                    pre_val = acc[i] + IN_WIDTH'($signed(bias[i]));
                        

                    // ---------------------------------------------------------
                    // ReLU + Clamp
                    // ---------------------------------------------------------

                    if (pre_val <= 0)
                        relu_r[i] <= '0;

                    else if (pre_val > RELU_MAX)
                        relu_r[i] <= RELU_MAX;

                    else
                        relu_r[i] <= pre_val;

                end
            end
        end
    end

    // =========================================================================
    // Stage 2
    // Shift + INT8 Saturation
    // =========================================================================

    always_ff @(posedge clk) begin

        if (!rst_n) begin

            valid_s2 <= 1'b0;

            for (int i = 0; i < N; i++) begin
                dout[i] <= '0;
            end

        end
        else begin

            valid_s2 <= valid_s1;

            if (valid_s1) begin

                for (int i = 0; i < N; i++) begin

                    logic signed [IN_WIDTH-1:0] shift_val;

                    // ---------------------------------------------------------
                    // Arithmetic Requant Shift
                    // ---------------------------------------------------------

                    if (SHIFT == 0)
                        shift_val = relu_r[i];
                    else
                        shift_val = relu_r[i] >>> SHIFT;

                    // ---------------------------------------------------------
                    // INT8 Saturation
                    // ---------------------------------------------------------

                    if (shift_val > 127)
                        dout[i] <= 8'sd127;

                    else if (shift_val < -128)
                        dout[i] <= -8'sd128;

                    else
                        dout[i] <= shift_val[7:0];

                end
            end
        end
    end

    // =========================================================================
    // Output Valid
    // =========================================================================

    assign valid = valid_s2;

endmodule
`timescale 1ns/1ps

import parameter_pkg::*;

module mac_unit #(
    parameter int IN_WIDTH  = FEAT_W,
    parameter int WT_WIDTH  = W_WIDTH,
    parameter int ACC_WIDTH = ACC_W
)(
    input  logic                         clk,
    input  logic                         rst_n,
    input  logic                         en,
    input  logic                         clear_acc,

    input  logic signed [IN_WIDTH-1:0]  a,
    input  logic signed [WT_WIDTH-1:0]  w,

    output logic signed [ACC_WIDTH-1:0] acc
);

    localparam int PROD_WIDTH = IN_WIDTH + WT_WIDTH;

    typedef logic signed [PROD_WIDTH-1:0] prod_t;
    typedef logic signed [ACC_WIDTH-1:0]  acc_t;

    prod_t prod_r;

    logic valid_r;
    logic clear_r;

    acc_t acc_r;

    // =========================================================
    // Stage 1 : Multiply Pipeline
    // =========================================================

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            prod_r  <= '0;
            valid_r <= 1'b0;
            clear_r <= 1'b0;
        end
        else begin
            valid_r <= en;

            if (en) begin
                prod_r  <= $signed(a) * $signed(w);
                clear_r <= clear_acc;
            end
        end
    end

    // =========================================================
    // Stage 2 : Accumulator
    // =========================================================

    (* use_dsp = "yes" *)
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            acc_r <= '0;
        end
        else if (valid_r) begin

            if (clear_r)
                acc_r <= acc_t'(prod_r);

            else
                acc_r <= acc_r + acc_t'(prod_r);
        end
    end

    assign acc = acc_r;

endmodule
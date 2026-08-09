`timescale 1ns/1ps

module layer3_mlp
    import parameter_pkg::*;
#(

    parameter int IN_DIM      = L3_IN,
    parameter int OUT_DIM     = L3_OUT,

    parameter int PAR         = PARALLELISM,

    parameter int IN_WIDTH    = FEAT_W,
    parameter int WT_WIDTH    = W_WIDTH,
    parameter int ACC_WIDTH   = ACC_W,
    parameter int BIAS_WIDTH  = BIAS_W,

    parameter int WBRAM_DEPTH = L3_WBRAM_DEPTH,
    parameter int BBRAM_DEPTH = L3_BBRAM_DEPTH,

    parameter string W_MEM_FILE =
        "layer3_weights_quantized3.mem",

    parameter string B_MEM_FILE =
        "layer3_bias_quantized3.mem"

)(

    input logic clk,
    input logic rst_n,
    input logic start,

    input logic signed [7:0]
        in_vec [IN_DIM-1:0],

    output logic signed [ACC_WIDTH-1:0]
        logits [OUT_DIM-1:0],

    output logic done

);

    // =========================================================================
    // PARAMETERS
    // =========================================================================

    localparam int STEP_W =
        $clog2(IN_DIM + 2);

    typedef enum logic [2:0] {

        S_IDLE,
        S_PREFETCH,
        S_RUN,
        S_WAIT,
        S_BIAS_PREFETCH,
        S_BIAS_CAPTURE,
        S_DONE

    } state_t;

    state_t state;

    logic [STEP_W-1:0]
        step;

    logic first_run;

    // =========================================================================
    // WEIGHT BRAMS
    // =========================================================================

    logic wt_en;

    logic [$clog2(WBRAM_DEPTH)-1:0]
        wt_addr;

    logic signed [WT_WIDTH-1:0]
        wt_data [PAR-1:0];

    for(genvar k=0;k<PAR;k++) begin : gen_weight_brams

        weight_bram #(
            .DATA_WIDTH(WT_WIDTH),
            .DEPTH(WBRAM_DEPTH),
            .MEM_FILE(W_MEM_FILE)
        ) wbram (

            .clk(clk),
            .en(wt_en),

            .addr(wt_addr + (k * IN_DIM)),

            .dout(wt_data[k])

        );

    end

    // =========================================================================
    // MAC ARRAY
    // =========================================================================

    logic mac_start;
    logic mac_done;

    logic signed [ACC_WIDTH-1:0]
        mac_out [PAR-1:0];

    mac_array #(
        .PARALLELISM(PAR),
        .IN_DIM(IN_DIM),
        .IN_WIDTH(IN_WIDTH),
        .WT_WIDTH(WT_WIDTH),
        .ACC_WIDTH(ACC_WIDTH),
        .ACTIVE_MACS(OUT_DIM)
    ) mac (

        .clk(clk),
        .rst_n(rst_n),

        .start(mac_start),

        .in_vec(in_vec),

        .wt_data(wt_data),

        .acc_out(mac_out),

        .done(mac_done)

    );

    // =========================================================================
    // BIAS BRAMS
    // =========================================================================

    logic bias_en;

    logic [$clog2(BBRAM_DEPTH)-1:0]
        bias_addr;

    logic signed [BIAS_WIDTH-1:0]
        bias_data [PAR-1:0];

    for(genvar k=0;k<PAR;k++) begin : gen_bias_brams

        bias_bram #(
            .DEPTH(BBRAM_DEPTH),
            .MEM_FILE(B_MEM_FILE)
        ) bbram (

            .clk(clk),
            .en(bias_en),

            .addr(bias_addr + k),

            .dout(bias_data[k])

        );

    end

    // =========================================================================
    // FSM
    // =========================================================================

    always_ff @(posedge clk or negedge rst_n) begin

        if(!rst_n) begin

            state <= S_IDLE;

            done <= 1'b0;

            wt_en     <= 1'b0;
            bias_en   <= 1'b0;
            mac_start <= 1'b0;

            wt_addr   <= '0;
            bias_addr <= '0;

            step <= '0;

            first_run <= 1'b0;

            for(int i=0;i<OUT_DIM;i++)
                logits[i] <= '0;

        end
        else begin

            done <= 1'b0;

            mac_start <= 1'b0;

            case(state)

                // =============================================================
                // IDLE
                // =============================================================

                S_IDLE: begin

                    wt_en   <= 1'b0;
                    bias_en <= 1'b0;

                    if(start) begin

                        wt_en <= 1'b1;

                        wt_addr <= '0;

                        step <= '0;

                        first_run <= 1'b1;

                        state <= S_PREFETCH;

                    end

                end

                // =============================================================
                // PREFETCH
                // =============================================================

                S_PREFETCH: begin

                    state <= S_RUN;

                end

                // =============================================================
                // RUN
                // =============================================================

                S_RUN: begin

                    if(first_run) begin

                        mac_start <= 1'b1;

                        first_run <= 1'b0;

                    end

                    if(step < (IN_DIM-1)) begin

                        step <= step + 1'b1;

                        wt_addr <= step + 1'b1;

                    end
                    else begin

                        wt_en <= 1'b0;

                        state <= S_WAIT;

                    end

                end

                // =============================================================
                // WAIT
                // =============================================================

                S_WAIT: begin

                    if(mac_done) begin

                        bias_en <= 1'b1;

                        bias_addr <= '0;

                        state <= S_BIAS_PREFETCH;

                    end

                end

                // =============================================================
                // BIAS PREFETCH
                // =============================================================

                S_BIAS_PREFETCH: begin

                    state <= S_BIAS_CAPTURE;

                end

                // =============================================================
                // BIAS + LOGITS
                // =============================================================

                S_BIAS_CAPTURE: begin

                    for(int i=0;i<OUT_DIM;i++) begin

                        logits[i]
                            <= mac_out[i]
                            + ACC_WIDTH'(signed'(bias_data[i]));

                    end

                    bias_en <= 1'b0;

                    state <= S_DONE;

                end

                // =============================================================
                // DONE
                // =============================================================

                S_DONE: begin

                    done <= 1'b1;

                    state <= S_IDLE;

                end

                default: begin

                    state <= S_IDLE;

                end

            endcase

        end

    end

endmodule
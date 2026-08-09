`timescale 1ns/1ps

module layer2_mlp
    import parameter_pkg::*;
#(

    parameter int IN_DIM      = L2_IN,
    parameter int OUT_DIM     = L2_OUT,

    parameter int PAR         = PARALLELISM,

    parameter int SHIFT       = L2_SHIFT,

    parameter int IN_WIDTH    = FEAT_W,
    parameter int WT_WIDTH    = W_WIDTH,
    parameter int ACC_WIDTH   = ACC_W,
    parameter int BIAS_WIDTH  = BIAS_W,

    parameter int WBRAM_DEPTH = L2_WBRAM_DEPTH,
    parameter int BBRAM_DEPTH = L2_BBRAM_DEPTH,

    parameter string W_MEM_FILE =
        "layer2_weights_quantized3.mem",

    parameter string B_MEM_FILE =
        "layer2_bias_quantized3.mem"

)(

    input logic clk,
    input logic rst_n,
    input logic start,

    input logic signed [7:0]
        in_vec [IN_DIM-1:0],

    output logic signed [7:0]
        act_out [OUT_DIM-1:0],

    output logic done

);

    localparam int N_GROUPS =
        (OUT_DIM + PAR - 1) / PAR;

    localparam int STEP_W =
        $clog2(
            ((OUT_DIM > IN_DIM) ? OUT_DIM : IN_DIM)
            + 2
        );

    localparam int GROUP_W =
        (N_GROUPS <= 1) ? 1 :
        $clog2(N_GROUPS);

    typedef enum logic [3:0] {

        S_IDLE,
        S_PREFETCH,
        S_RUN,
        S_WAIT,
        S_BIAS_PREFETCH,
        S_BIAS_CAPTURE,
        S_ACT_PREP,
        S_ACT

    } state_t;

    state_t state;

    logic [GROUP_W-1:0] group;
    logic [STEP_W-1:0]  step;

    logic first_run;

    // =========================================================================
    // Weight BRAM
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

            .addr(
                wt_addr + (k * IN_DIM)
            ),

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
    // Storage
    // =========================================================================

    logic signed [ACC_WIDTH-1:0]
        acc_arr [OUT_DIM-1:0];

    logic signed [BIAS_WIDTH-1:0]
        bias_arr [OUT_DIM-1:0];

    // =========================================================================
    // Bias BRAM
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
    // Activation
    // =========================================================================

    logic act_en;
    logic act_valid;

    logic signed [7:0]
        act_tmp [OUT_DIM-1:0];

    activation_relu_quant #(
        .N(OUT_DIM),
        .IN_WIDTH(ACC_WIDTH),
        .B_WIDTH(BIAS_WIDTH),
        .SHIFT(SHIFT)
    ) relu (

        .clk(clk),
        .rst_n(rst_n),

        .en(act_en),

        .acc(acc_arr),
        .bias(bias_arr),

        .dout(act_tmp),

        .valid(act_valid)

    );

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
            act_en    <= 1'b0;

            group <= '0;
            step  <= '0;

            wt_addr   <= '0;
            bias_addr <= '0;

            first_run <= 1'b0;

            for(int i=0;i<OUT_DIM;i++) begin

                acc_arr[i]  <= '0;
                bias_arr[i] <= '0;
                act_out[i]  <= '0;

            end

        end
        else begin

            done      <= 1'b0;
            mac_start <= 1'b0;
            act_en    <= 1'b0;

            case(state)

                S_IDLE: begin

                    wt_en   <= 1'b0;
                    bias_en <= 1'b0;

                    if(start) begin

                        group <= '0;
                        step  <= '0;

                        wt_en   <= 1'b1;
                        wt_addr <= '0;

                        first_run <= 1'b1;

                        state <= S_PREFETCH;

                    end

                end

                S_PREFETCH: begin

                    state <= S_RUN;

                end

                S_RUN: begin

                    if(first_run) begin

                        mac_start <= 1'b1;
                        first_run <= 1'b0;

                    end

                    if(step < (IN_DIM-1)) begin

                        step <= step + 1'b1;

                        wt_addr <=
                            (group * PAR * IN_DIM)
                            + (step + 1'b1);

                    end
                    else begin

                        wt_en <= 1'b0;

                        state <= S_WAIT;

                    end

                end

                S_WAIT: begin

                    if(mac_done) begin

                        for(int k=0;k<PAR;k++) begin

                            if((group*PAR + k) < OUT_DIM)

                                acc_arr[group*PAR + k]
                                    <= mac_out[k];

                        end

                        bias_en <= 1'b1;

                        bias_addr <= group * PAR;

                        state <= S_BIAS_PREFETCH;

                    end

                end

                S_BIAS_PREFETCH: begin

                    state <= S_BIAS_CAPTURE;

                end

                S_BIAS_CAPTURE: begin

                    for(int k=0;k<PAR;k++) begin

                        if((group*PAR + k) < OUT_DIM)

                            bias_arr[group*PAR + k]
                                <= bias_data[k];

                    end

                    bias_en <= 1'b0;

                    state <= S_ACT_PREP;

                end

                S_ACT_PREP: begin

                    act_en <= 1'b1;

                    state <= S_ACT;

                end

                S_ACT: begin

                    if(act_valid) begin

                        for(int i=0;i<OUT_DIM;i++)
                            act_out[i] <= act_tmp[i];

                        done <= 1'b1;

                        state <= S_IDLE;

                    end

                end

                default: begin

                    state <= S_IDLE;

                end

            endcase

        end

    end

endmodule
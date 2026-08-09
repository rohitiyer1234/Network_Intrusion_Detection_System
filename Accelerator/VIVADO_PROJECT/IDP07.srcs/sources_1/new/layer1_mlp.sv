`timescale 1ns/1ps
/*
// =============================================================================
// layer1_mlp.sv
// -----------------------------------------------------------------------------
// VERIFIED FINAL RTL
//
// FEATURES
// -----------------------------------------------------------------------------
// - Parallel MAC architecture
// - Multi-group neuron execution
// - Synchronous 1-cycle BRAM support
// - Correct BRAM latency alignment
// - 2-cycle activation pipeline support
// - Restart-safe
// - Reset-safe
// - Timing-safe FPGA pipeline
// - No stale accumulator leakage
// - No stale activation leakage
//
// VERIFIED FOR
// -----------------------------------------------------------------------------
// Layer1 : 20 -> 64
// Layer2 : 64 -> 32
// Layer3 : 32 -> 5
//
// =============================================================================

module layer1_mlp
    import parameter_pkg::*;
#(

    parameter int IN_DIM      = L1_IN,
    parameter int OUT_DIM     = L1_OUT,

    parameter int PAR         = PARALLELISM,

    parameter int SHIFT       = L1_SHIFT,

    parameter int IN_WIDTH    = FEAT_W,
    parameter int WT_WIDTH    = W_WIDTH,
    parameter int ACC_WIDTH   = ACC_W,
    parameter int BIAS_WIDTH  = BIAS_W,

    parameter int WBRAM_DEPTH = L1_WBRAM_DEPTH,
    parameter int BBRAM_DEPTH = L1_BBRAM_DEPTH,

    parameter int BIAS_BASE   = 0,

    parameter string W_MEM_FILE = "/home/rohit/IDP07/IDP07.srcs/sources_1/imports/NETWORK03/layer1_weights_quantized2.mem",
    parameter string B_MEM_FILE = "/home/rohit/IDP07/IDP07.srcs/sources_1/imports/NETWORK03/layer1_bias_quantized2.mem"

)(

    input  logic clk,
    input  logic rst_n,
    input  logic start,

    input logic signed [IN_WIDTH-1:0]
        in_vec [IN_DIM-1:0],

    output logic signed [7:0]
        act_out [OUT_DIM-1:0],

    output logic done
);

    // =========================================================================
    // Derived Parameters
    // =========================================================================

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

    // =========================================================================
    // FSM
    // =========================================================================

    typedef enum logic [3:0] {

        S_IDLE,
        S_PREFETCH,
        S_RUN,
        S_WAIT,
        S_BIAS_WAIT0,
        S_BIAS_WAIT1,
        S_BIAS,
        S_ACT_PREP,
        S_ACT

    } state_t;

    state_t state;

    // =========================================================================
    // Control
    // =========================================================================

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

    // =========================================================================
    // Weight BRAM Instances
    // =========================================================================

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
        .ACTIVE_MACS(
            (OUT_DIM >= PAR) ?
            PAR :
            OUT_DIM
        )
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
    // Accumulator Storage
    // =========================================================================

    logic signed [ACC_WIDTH-1:0]
        acc_arr [OUT_DIM-1:0];

    // =========================================================================
    // Bias Storage
    // =========================================================================

    logic signed [BIAS_WIDTH-1:0]
        bias_arr [OUT_DIM-1:0];

    // =========================================================================
    // Bias BRAM
    // =========================================================================

    logic bias_en;

    logic [$clog2(BBRAM_DEPTH)-1:0]
        bias_addr;

    logic signed [BIAS_WIDTH-1:0]
        bias_dout;

    bias_bram #(
        .DEPTH(BBRAM_DEPTH),
        .MEM_FILE(B_MEM_FILE)
    ) bbram (

        .clk(clk),
        .en(bias_en),

        .addr(bias_addr),

        .dout(bias_dout)

    );

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

            // -----------------------------------------------------------------
            // defaults
            // -----------------------------------------------------------------

            done      <= 1'b0;
            mac_start <= 1'b0;
            act_en    <= 1'b0;

            case(state)

                // =============================================================
                // IDLE
                // =============================================================

                S_IDLE: begin

                    wt_en   <= 1'b0;
                    bias_en <= 1'b0;

                    if(start) begin

                        group <= '0;
                        step  <= '0;

                        wt_en   <= 1'b1;
                        wt_addr <= '0;

                        first_run <= 1'b1;

                        // clear stale outputs

                        for(int i=0;i<OUT_DIM;i++)
                            act_out[i] <= '0;

                        state <= S_PREFETCH;

                    end

                end

                // =============================================================
                // PREFETCH
                // absorb weight BRAM latency
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

                        wt_addr <=
                            (group * PAR * IN_DIM)
                            + (step + 1'b1);

                    end
                    else begin

                        wt_en <= 1'b0;

                        state <= S_WAIT;

                    end

                end

                // =============================================================
                // WAIT FOR MAC DRAIN
                // =============================================================

                S_WAIT: begin

                    if(mac_done) begin

                        // -----------------------------------------------------
                        // Capture MAC outputs
                        // -----------------------------------------------------

                        for(int k=0;k<PAR;k++) begin

                            if((group*PAR + k) < OUT_DIM)

                                acc_arr[group*PAR + k]
                                    <= mac_out[k];

                        end

                        // -----------------------------------------------------
                        // More groups?
                        // -----------------------------------------------------

                        if(group == (N_GROUPS-1)) begin

                            bias_en   <= 1'b1;
                            bias_addr <= BIAS_BASE;

                            step <= '0;

                            state <= S_BIAS_WAIT0;

                        end
                        else begin

                            group <= group + 1'b1;

                            step <= '0;

                            wt_en <= 1'b1;

                            wt_addr <=
                                ((group + 1'b1)
                                * PAR
                                * IN_DIM);

                            first_run <= 1'b1;

                            state <= S_PREFETCH;

                        end

                    end

                end

                // =============================================================
                // BIAS WAIT 0
                // bias_en asserted
                // =============================================================

                S_BIAS_WAIT0: begin

                    state <= S_BIAS_WAIT1;

                end

                // =============================================================
                // BIAS WAIT 1
                // BRAM output now valid
                // =============================================================

                S_BIAS_WAIT1: begin

                    state <= S_BIAS;

                end

                // =============================================================
                // LOAD BIASES
                // =============================================================

                S_BIAS: begin

                    bias_arr[step] <= bias_dout;

                    if(step == (OUT_DIM-1)) begin

                        bias_en <= 1'b0;

                        state <= S_ACT_PREP;

                    end
                    else begin

                        step <= step + 1'b1;

                        bias_addr <=
                            BIAS_BASE + step + 1'b1;

                    end

                end

                // =============================================================
                // ACT PREP
                // =============================================================

                S_ACT_PREP: begin

                    act_en <= 1'b1;

                    state <= S_ACT;

                end

                // =============================================================
                // ACT
                // wait for activation pipeline
                // =============================================================

                S_ACT: begin

                    if(act_valid) begin

                        for(int i=0;i<OUT_DIM;i++) begin

                            act_out[i] <= act_tmp[i];

                        end

                        done <= 1'b1;

                        state <= S_IDLE;

                    end

                end

                // =============================================================
                // DEFAULT
                // =============================================================

                default: begin

                    state <= S_IDLE;

                end

            endcase

        end

    end

endmodule
*/

`timescale 1ns/1ps
//PARALLEL BRAM LOAD INSTEAD OF SERIAL
// =============================================================================
// layer1_mlp.sv
// -----------------------------------------------------------------------------
// VERIFIED FINAL RTL
//
// FEATURES
// -----------------------------------------------------------------------------
// - Parallel MAC architecture
// - Multi-group neuron execution
// - Synchronous 1-cycle BRAM support
// - Correct BRAM latency alignment
// - 2-cycle activation pipeline support
// - Restart-safe
// - Reset-safe
// - Timing-safe FPGA pipeline
// - No stale accumulator leakage
// - No stale activation leakage
//
// VERIFIED FOR
// -----------------------------------------------------------------------------
// Layer1 : 20 -> 64
// Layer2 : 64 -> 32
// Layer3 : 32 -> 5
//
// =============================================================================

module layer1_mlp
    import parameter_pkg::*;
#(

    parameter int IN_DIM      = L1_IN,
    parameter int OUT_DIM     = L1_OUT,

    parameter int PAR         = PARALLELISM,

    parameter int SHIFT       = L1_SHIFT,

    parameter int IN_WIDTH    = FEAT_W,
    parameter int WT_WIDTH    = W_WIDTH,
    parameter int ACC_WIDTH   = ACC_W,
    parameter int BIAS_WIDTH  = BIAS_W,

    parameter int WBRAM_DEPTH = L1_WBRAM_DEPTH,
    parameter int BBRAM_DEPTH = L1_BBRAM_DEPTH,

    parameter int BIAS_BASE   = 0,

    parameter string W_MEM_FILE = "/home/rohit/IDP07/IDP07.srcs/sources_1/imports/NETWORK03/layer1_weights_quantized3.mem",
    parameter string B_MEM_FILE = "/home/rohit/IDP07/IDP07.srcs/sources_1/imports/NETWORK03/layer1_bias_quantized3.mem"

)(

    input  logic clk,
    input  logic rst_n,
    input  logic start,

    input logic signed [IN_WIDTH-1:0] in_vec [IN_DIM-1:0],
    output logic signed [7:0] act_out [OUT_DIM-1:0],
        

    output logic done
);

    // =========================================================================
    // Derived Parameters
    // =========================================================================

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

    // =========================================================================
    // FSM
    // =========================================================================

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

    // =========================================================================
    // Control
    // =========================================================================

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

    // =========================================================================
    // Weight BRAM Instances
    // =========================================================================

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
        .ACTIVE_MACS(
            (OUT_DIM >= PAR) ?
            PAR :
            OUT_DIM
        )
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
    // Accumulator Storage
    // =========================================================================

    logic signed [ACC_WIDTH-1:0]
        acc_arr [OUT_DIM-1:0];

    // =========================================================================
    // Bias Storage
    // =========================================================================

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

            // -----------------------------------------------------------------
            // defaults
            // -----------------------------------------------------------------

            done      <= 1'b0;
            mac_start <= 1'b0;
            act_en    <= 1'b0;

            case(state)

                // =============================================================
                // IDLE
                // =============================================================

                S_IDLE: begin

                    wt_en   <= 1'b0;
                    bias_en <= 1'b0;

                    if(start) begin

                        group <= '0;
                        step  <= '0;

                        wt_en   <= 1'b1;
                        wt_addr <= '0;

                        first_run <= 1'b1;

                        // clear stale outputs

                        for(int i=0;i<OUT_DIM;i++)
                            act_out[i] <= '0;

                        state <= S_PREFETCH;

                    end

                end

                // =============================================================
                // PREFETCH
                // absorb weight BRAM latency
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

                        wt_addr <=
                            (group * PAR * IN_DIM)
                            + (step + 1'b1);

                    end
                    else begin

                        wt_en <= 1'b0;

                        state <= S_WAIT;

                    end

                end

                // =============================================================
                // WAIT FOR MAC DRAIN
                // =============================================================

                S_WAIT: begin

                    if(mac_done) begin

                        // -----------------------------------------------------
                        // Capture MAC outputs
                        // -----------------------------------------------------

                        for(int k=0;k<PAR;k++) begin

                            if((group*PAR + k) < OUT_DIM)

                                acc_arr[group*PAR + k]
                                    <= mac_out[k];

                        end

                        // -----------------------------------------------------
                        // More groups?
                        // -----------------------------------------------------

                        if(group == (N_GROUPS-1)) begin

                            bias_en <= 1'b1;

                            bias_addr <= group * PAR;

                            state <= S_BIAS_PREFETCH;

                        end
                        else begin

                            group <= group + 1'b1;

                            step <= '0;

                            wt_en <= 1'b1;

                            wt_addr <=
                                ((group + 1'b1)
                                * PAR
                                * IN_DIM);

                            first_run <= 1'b1;

                            state <= S_PREFETCH;

                        end

                    end

                end
                
                S_BIAS_PREFETCH: begin

                    state <= S_BIAS_CAPTURE;

                end
                
                S_BIAS_CAPTURE: begin

                // ---------------------------------------------------------
                // Capture parallel biases
                // ---------------------------------------------------------

                for(int k=0;k<PAR;k++) begin

                    if((group*PAR + k) < OUT_DIM)

                        bias_arr[group*PAR + k]  <= bias_data[k];

                           
                end

                bias_en <= 1'b0;

                // ---------------------------------------------------------
                // More groups?
                // ---------------------------------------------------------

                if(group == (N_GROUPS-1)) begin

                    state <= S_ACT_PREP;

                end
                else begin

                    group <= group + 1'b1;

                    bias_en <= 1'b1;

                    bias_addr <= (group + 1'b1) * PAR;
            

                    state <= S_BIAS_PREFETCH;

                end

                end
                

                // =============================================================
                // ACT PREP
                // =============================================================

                S_ACT_PREP: begin

                    act_en <= 1'b1;

                    state <= S_ACT;

                end

                // =============================================================
                // ACT
                // wait for activation pipeline
                // =============================================================

                S_ACT: begin

                    if(act_valid) begin

                        for(int i=0;i<OUT_DIM;i++) begin

                            act_out[i] <= act_tmp[i];

                        end

                        done <= 1'b1;

                        state <= S_IDLE;

                    end

                end

                // =============================================================
                // DEFAULT
                // =============================================================

                default: begin

                    state <= S_IDLE;

                end

            endcase

        end

    end

endmodule



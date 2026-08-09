`timescale 1ns/1ps

module mlp_top
    import parameter_pkg::*;
#(

    parameter string L1_W_MEM =
        "layer1_weights_quantized3.mem",

    parameter string L2_W_MEM =
        "layer2_weights_quantized3.mem",

    parameter string L3_W_MEM =
        "layer3_weights_quantized3.mem",

    parameter string L1_BIAS_MEM =
        "layer1_bias_quantized3.mem",

    parameter string L2_BIAS_MEM =
        "layer2_bias_quantized3.mem",

    parameter string L3_BIAS_MEM =
        "layer3_bias_quantized3.mem"

)(

    input logic aclk,
    input logic aresetn,

    // =========================================================================
    // AXI LITE
    // =========================================================================

    input logic [ALIT_AW-1:0] s_axi_awaddr,
    input logic s_axi_awvalid,
    output logic s_axi_awready,

    input logic [ALIT_W-1:0] s_axi_wdata,
    input logic [(ALIT_W/8)-1:0] s_axi_wstrb,
    input logic s_axi_wvalid,
    output logic s_axi_wready,

    output logic [1:0] s_axi_bresp,
    output logic s_axi_bvalid,
    input logic s_axi_bready,

    input logic [ALIT_AW-1:0] s_axi_araddr,
    input logic s_axi_arvalid,
    output logic s_axi_arready,

    output logic [ALIT_W-1:0] s_axi_rdata,
    output logic [1:0] s_axi_rresp,
    output logic s_axi_rvalid,
    input logic s_axi_rready,

    // =========================================================================
    // AXIS INPUT
    // =========================================================================

    input logic [AXIS_W-1:0] s_axis_tdata,
    input logic s_axis_tvalid,
    output logic s_axis_tready,
    input logic s_axis_tlast,

    // =========================================================================
    // AXIS OUTPUT
    // =========================================================================

    output logic [AXIS_W-1:0] m_axis_tdata,
    output logic m_axis_tvalid,
    input logic m_axis_tready,
    output logic m_axis_tlast

);

    // =========================================================================
    // RESET SYNCHRONIZER
    // =========================================================================

    logic rst_n_r1;
    logic rst_n_r2;

    always_ff @(posedge aclk or negedge aresetn) begin

        if(!aresetn) begin

            rst_n_r1 <= 1'b0;
            rst_n_r2 <= 1'b0;

        end
        else begin

            rst_n_r1 <= 1'b1;
            rst_n_r2 <= rst_n_r1;

        end

    end

    logic rst_n;

    assign rst_n = rst_n_r2;

    // =========================================================================
    // AXI-LITE
    // =========================================================================

    logic wr_en;
    logic [3:0] wr_addr;
    logic [31:0] wr_data;

    logic rd_en;
    logic [3:0] rd_addr;
    logic [31:0] rd_data;

    // =========================================================================
    // CONTROL
    // =========================================================================

    logic ctrl_start;
    logic ctrl_reset;

    logic status_idle;
    logic status_busy;
    logic status_done;
    logic status_error;

    logic [31:0] inf_counter;
    logic [31:0] latency_cycles;

    // =========================================================================
    // FEATURE BUFFER
    // =========================================================================

    logic fb_write_en;

    logic signed [FEAT_W-1:0]
        fb_write_data;

    logic signed [FEAT_W-1:0]
        features [L1_IN-1:0];

    logic feat_full;

    logic feat_clear;

    // =========================================================================
    // STREAM STATUS
    // =========================================================================

    logic packet_error;

    // =========================================================================
    // LAYER CONTROL
    // =========================================================================

    logic l1_start;
    logic l2_start;
    logic l3_start;

    logic l1_done;
    logic l2_done;
    logic l3_done;

    // =========================================================================
    // LAYER DATA
    // =========================================================================

    logic signed [7:0]
        l1_act [L1_OUT-1:0];

    logic signed [7:0]
        l2_act [L2_OUT-1:0];

    logic signed [ACC_W-1:0]
        l3_logits [L3_OUT-1:0];

    // =========================================================================
    // OUTPUT
    // =========================================================================

    logic send_req;

    logic tx_done;

    logic logit_we;

    logic signed [ACC_W-1:0]
        out_logits [L3_OUT-1:0];

    // =========================================================================
    // GLOBAL ERROR
    // =========================================================================

    logic global_error;

    assign global_error =
        packet_error
        ||
        status_error;

    // =========================================================================
    // AXI LITE SLAVE
    // =========================================================================

    axi_lite_slave u_axi_lite (

        .s_axi_aclk(aclk),
        .s_axi_aresetn(rst_n),

        .s_axi_awaddr(s_axi_awaddr),
        .s_axi_awvalid(s_axi_awvalid),
        .s_axi_awready(s_axi_awready),

        .s_axi_wdata(s_axi_wdata),
        .s_axi_wstrb(s_axi_wstrb),
        .s_axi_wvalid(s_axi_wvalid),
        .s_axi_wready(s_axi_wready),

        .s_axi_bresp(s_axi_bresp),
        .s_axi_bvalid(s_axi_bvalid),
        .s_axi_bready(s_axi_bready),

        .s_axi_araddr(s_axi_araddr),
        .s_axi_arvalid(s_axi_arvalid),
        .s_axi_arready(s_axi_arready),

        .s_axi_rdata(s_axi_rdata),
        .s_axi_rresp(s_axi_rresp),
        .s_axi_rvalid(s_axi_rvalid),
        .s_axi_rready(s_axi_rready),

        .wr_en(wr_en),
        .wr_addr(wr_addr),
        .wr_data(wr_data),

        .rd_en(rd_en),
        .rd_addr(rd_addr),
        .rd_data(rd_data)

    );

    // =========================================================================
    // REGISTER BANK
    // =========================================================================

    register_bank u_regbank (

        .clk(aclk),
        .rst_n(rst_n),

        .wr_en(wr_en),
        .wr_addr(wr_addr),
        .wr_data(wr_data),

        .rd_en(rd_en),
        .rd_addr(rd_addr),
        .rd_data(rd_data),

        .ctrl_start(ctrl_start),
        .ctrl_reset(ctrl_reset),

        .status_idle(status_idle),
        .status_busy(status_busy),
        .status_done(status_done),

        .inf_counter(inf_counter),
        .latency_cycles(latency_cycles),

        .logit_we(logit_we),
        .logits(out_logits)

    );

    // =========================================================================
    // AXIS SLAVE
    // =========================================================================

    axi_stream_slave u_axis_slave (

        .clk(aclk),
        .rst_n(rst_n),

        .s_axis_tdata(s_axis_tdata),
        .s_axis_tvalid(s_axis_tvalid),
        .s_axis_tready(s_axis_tready),
        .s_axis_tlast(s_axis_tlast),

        .fb_write_en(fb_write_en),
        .fb_write_data(fb_write_data),

        .frame_done(),

        .fb_full(feat_full),

        .packet_error(packet_error)

    );

    // =========================================================================
    // FEATURE BUFFER
    // =========================================================================

    feature_buffer u_feat_buf (

        .clk(aclk),
        .rst_n(rst_n),

        .clear(
            feat_clear
            ||
            global_error
        ),

        .write_en(fb_write_en),
        .write_data(fb_write_data),

        .features(features),

        .full(feat_full)

    );

    // =========================================================================
    // CONTROL FSM
    // =========================================================================

    control_fsm u_fsm (

        .clk(aclk),
        .rst_n(rst_n),

        .ctrl_start(ctrl_start),
        .ctrl_reset(
            ctrl_reset
            ||
            global_error
        ),

        .feat_full(feat_full),
        .feat_clear(feat_clear),

        .l1_start(l1_start),
        .l1_done(l1_done),

        .l2_start(l2_start),
        .l2_done(l2_done),

        .l3_start(l3_start),
        .l3_done(l3_done),

        .send_req(send_req),
        .tx_done(tx_done),

        .logit_we(logit_we),

        .l3_logits(l3_logits),
        .out_logits(out_logits),

        .status_idle(status_idle),
        .status_busy(status_busy),
        .status_done(status_done),
        .status_error(status_error),

        .inf_counter(inf_counter),
        .latency_cycles(latency_cycles)

    );

    // =========================================================================
    // LAYER 1
    // =========================================================================

    layer1_mlp #(

        .W_MEM_FILE(L1_W_MEM),
        .B_MEM_FILE(L1_BIAS_MEM)

    ) u_layer1 (

        .clk(aclk),
        .rst_n(rst_n),

        .start(l1_start),

        .in_vec(features),

        .act_out(l1_act),

        .done(l1_done)

    );

    // =========================================================================
    // LAYER 2
    // =========================================================================

    layer2_mlp #(

        .W_MEM_FILE(L2_W_MEM),
        .B_MEM_FILE(L2_BIAS_MEM)

    ) u_layer2 (

        .clk(aclk),
        .rst_n(rst_n),

        .start(l2_start),

        .in_vec(l1_act),

        .act_out(l2_act),

        .done(l2_done)

    );

    // =========================================================================
    // LAYER 3
    // =========================================================================

    layer3_mlp #(

        .W_MEM_FILE(L3_W_MEM),
        .B_MEM_FILE(L3_BIAS_MEM)

    ) u_layer3 (

        .clk(aclk),
        .rst_n(rst_n),

        .start(l3_start),

        .in_vec(l2_act),

        .logits(l3_logits),

        .done(l3_done)

    );

    // =========================================================================
    // AXIS MASTER
    // =========================================================================

    axi_stream_master u_axis_master (

        .clk(aclk),
        .rst_n(rst_n),

        .send_req(send_req),

        .logits(out_logits),

        .m_axis_tdata(m_axis_tdata),
        .m_axis_tvalid(m_axis_tvalid),
        .m_axis_tready(m_axis_tready),
        .m_axis_tlast(m_axis_tlast),

        .busy(),

        .tx_done(tx_done)

    );

endmodule
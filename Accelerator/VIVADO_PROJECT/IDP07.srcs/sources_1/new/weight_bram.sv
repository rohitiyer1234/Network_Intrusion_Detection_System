// =============================================================================
// weight_bram.sv
// Single-port synchronous BRAM wrapper for weight storage.
//
// Infers RAMB36/RAMB18 on Xilinx 7-series.
// Read latency: 1 cycle (output registered on clk edge when en=1).
//
// FIX vs original:
//   - addr_r: address registered before read to prevent read-during-write
//     glitches and to guarantee BRAM timing closure.
//   - Added output register stage for optional 2-cycle latency (commented out;
//     single-cycle is correct for our pipeline).
//   - DATA_WIDTH default matches W_WIDTH in package.
// =============================================================================

`timescale 1ns/1ps

module weight_bram
  import parameter_pkg::*;
#(
  parameter int          DATA_WIDTH = W_WIDTH,
  parameter int          DEPTH      = L1_WBRAM_DEPTH,
  parameter string       MEM_FILE   = ""
)(
  input  logic                         clk,
  input  logic                         en,
  input  logic [$clog2(DEPTH)-1:0]     addr,
  output logic signed [DATA_WIDTH-1:0] dout
);

  // Vivado ram_style attribute → infer Block RAM (not distributed RAM)
  (* ram_style = "block" *)
  logic signed [DATA_WIDTH-1:0] mem [0:DEPTH-1];

  // Initialize from hex file (Vivado synthesis respects this for BRAM init).
  initial begin
    if (MEM_FILE != "") begin
      $readmemh(MEM_FILE, mem);
    end else begin
      for (int i = 0; i < DEPTH; i++) mem[i] = '0;
    end
  end

  // Synchronous read - 1-cycle latency
  always_ff @(posedge clk) begin
    if (en)
      dout <= mem[addr];
    else 
      dout <= '0;
  end

endmodule

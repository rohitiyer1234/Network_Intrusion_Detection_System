`timescale 1ns/1ps

module bias_bram
  import parameter_pkg::*;
#(
  parameter int DEPTH = 64,
  parameter string MEM_FILE = ""
)(
  input  logic clk,
  input  logic en,

  input  logic [$clog2(DEPTH)-1:0] addr,

  output logic signed [BIAS_W-1:0] dout
);

  // ===========================================================================
  // MEMORY
  // ===========================================================================
  (* ram_style = "block" *)
  logic signed [BIAS_W-1:0] mem [0:DEPTH-1];

  // ===========================================================================
  // INIT
  // ===========================================================================
  initial begin

    if (MEM_FILE != "") begin
      $readmemh(MEM_FILE, mem);
    end else begin
      for (int i = 0; i < DEPTH; i++)
        mem[i] = '0;
    end

  end

  // ===========================================================================
  // SYNC READ
  // ===========================================================================
  always_ff @(posedge clk) begin

    if (en)
      dout <= mem[addr];
    else 
      dout <= '0;

  end

endmodule
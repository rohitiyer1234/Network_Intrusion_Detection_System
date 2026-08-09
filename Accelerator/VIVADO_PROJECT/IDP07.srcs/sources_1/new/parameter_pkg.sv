package parameter_pkg;

  // ===========================================================================
  // AXI
  // ===========================================================================
  localparam int AXIS_W = 64;

  localparam int ALIT_AW = 32;
  localparam int ALIT_W  = 32;

  // ===========================================================================
  // NETWORK ARCHITECTURE
  // ===========================================================================
  localparam int L1_IN  = 20;
  localparam int L1_OUT = 64;

  localparam int L2_IN  = 64;
  localparam int L2_OUT = 32;

  localparam int L3_IN  = 32;
  localparam int L3_OUT = 3;

  // ===========================================================================
  // DATA WIDTHS
  // ===========================================================================
  localparam int FEAT_W = 8;     // int8 input features
  localparam int W_WIDTH = 8;    // int8 weights
  localparam int BIAS_W  = 16;   // int16 bias
  localparam int ACC_W   = 32;   // int32 accumulation/logits

  // ===========================================================================
  // PARALLELISM
  // ===========================================================================
  localparam int PARALLELISM = 32;

  // ===========================================================================
  // WEIGHT BRAM DEPTHS
  // ===========================================================================
  localparam int L1_WBRAM_DEPTH = L1_IN * L1_OUT; // 20*64 = 1280
  localparam int L2_WBRAM_DEPTH = L2_IN * L2_OUT; // 64*32 = 2048
  localparam int L3_WBRAM_DEPTH = L3_IN * L3_OUT; // 32*3  = 96

  // ===========================================================================
  // BIAS BRAM DEPTHS
  // ===========================================================================
  localparam int L1_BBRAM_DEPTH = L1_OUT; // 64
  localparam int L2_BBRAM_DEPTH = L2_OUT; // 32
  localparam int L3_BBRAM_DEPTH = L3_OUT; // 3

  // ===========================================================================
  // REQUANTIZATION SHIFTS
  // Must match QAT training
  // ===========================================================================
  localparam int L1_SHIFT = 0;
  localparam int L2_SHIFT = 0;

  // ===========================================================================
  // CONTROL REGISTER BITS
  // ===========================================================================
  localparam int CTRL_START_BIT = 0;
  localparam int CTRL_RESET_BIT = 1;
  
  localparam int MAC_PIPE_DEPTH = 2;
endpackage
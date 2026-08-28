// Deterministic repair-free materialization of an endpoint-complete ELDA object.
module elda_sample(boundary_in_0, boundary_in_1, boundary_in_2, source_net_3, source_net_4, source_net_5);
  input boundary_in_0;
  input boundary_in_1;
  input boundary_in_2;
  output source_net_3;
  output source_net_4;
  output source_net_5;
  BUF_X1 u_0(.A(boundary_in_0), .Z(source_net_3));
  BUF_X1 u_1(.A(boundary_in_1), .Z(source_net_4));
  BUF_X1 u_2(.A(boundary_in_2), .Z(source_net_5));
endmodule

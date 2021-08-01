function [output_image] = rgb2opponent(input_image)
% converts an RGB image into opponent color space

R  = input_image(:,:,1);
G  = input_image(:,:,2);
B  = input_image(:,:,3);

R_op = (R-G)./sqrt(2);
G_op = (R+G-2*B)./sqrt(6);
B_op = (R+G+B)./sqrt(3);

output_image = input_image;
output_image(:,:,1) = R_op;
output_image(:,:,2) = G_op;
output_image(:,:,3) = B_op;


end


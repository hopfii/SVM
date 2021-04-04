%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright by Hans-Georg Beyer (HGB)
% For teaching use only! It is not allowed to use 
% this program without written permission by HGB 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% original modified to include option to set color
%
function ContourPlot_w_color(fkt_name, xl, xu, yl, yu, n_pts, n_contours, color, zmin, zmax)
% fkt_name: name (string) of function f(x,y) to be plotted
% xl: lower x-value, xu: upper x-value
% yl: lower y-value, yu: upper y-value
% n_pts+1: number of data points sampled in each direction
% n_contours: number of contour lines
% zmin: optional value of minimal f-contour
% zmax: optional value of maximal f-contour
%
  z = zeros(n_pts+1,n_pts+1);
  x = (xl:(xu-xl)/n_pts:xu);
  y = (yl:(yu-yl)/n_pts:yu);
  for j=1:n_pts+1
    for i=1:n_pts+1
      z(j,i) = feval(fkt_name, x(i), y(j));
    end
  end
  if ( nargin() ~= 9 )
    zmin = min(min(z));
    zmax = max(max(z));
  end
  if (n_contours <= 1)
    vn = [(zmax+zmin)/2, (zmax+zmin)/2];
  else
    deltaz = (zmax-zmin)/(n_contours-1);
    vn = (zmin:deltaz:zmax);
  end
  contour(x, y, z, vn, color);
end

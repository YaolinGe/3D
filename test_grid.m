clear;close all;clc;

nx = 20;
ny = 20;
dx = 1;
dy = dx * sin(pi/3);
x = 0:dx:nx;
y = 0:dy:ny;
[xx, yy] = meshgrid(x, y);
plot(xx, yy, 'k.')


xx_2 = xx;
xx_2(1:2:end, :) = xx_2(1:2:end, :) + dx / 2;
yy_2 = yy;
plot(xx_2, yy_2, 'ko')





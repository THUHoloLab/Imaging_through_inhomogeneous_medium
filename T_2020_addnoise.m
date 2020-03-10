close all;clear all;clc;
addpath('./Functions');
%% Paprameters (1)
nx=64; ny=64;nz=2; % data size
lambda=0.633;  % wavelength (um)
detector_size = 5;  % pixel pitch (um)
sensor_size=nx*detector_size;  % detector size (um)
deltaZ = 1.5*1000;  % axial spacing (um)
offsetZ = 0*1000;  % distance from detector to first reconstructed plane (um)
deltaX = detector_size;  deltaY = detector_size;
Nx=nx; Ny=ny*nz*2; Nz=1;     
%% Object generation (2)
f=zeros(nx,ny,nz);
pic_T = zeros(64,64);
pic_T(30,30:34) = 1; pic_T(31:34,32) = 1; % make a pattern of letter "T" 
f(:,:,2)= pic_T;
figure;imagesc(plotdatacube(abs(f)));title('3D object');axis image;drawnow;
axis off; colormap(hot); colorbar;
%% Propagation kernel (3)
[ Phase3D Pupil ]=MyMakingPhase3D(nx,ny,nz,lambda,deltaX,deltaY,deltaZ,offsetZ,sensor_size);
figure;imagesc( plotdatacube(angle(Phase3D)) );title('Phase of kernel');axis image;drawnow;
axis off; colormap(hot); colorbar;
E0 = ones(nx,ny);  % illumination of plane wave
E0 = E0.*exp(i.*pi*0.0);
E = MyFieldsPropagation( E0,nx,ny,nz,Phase3D,Pupil );  % propagation of illumination light
%% Field measurement and backpropagation (4)
cEs=zeros(nx,ny,nz);
S  = angular_spectrum_function( f(:,:,2) , detector_size , lambda ,1*deltaZ + offsetZ);
s=(S+1).*conj(S+1); 
s = s - mean( s(:) ) ;
figure; subplot(1,2,1),imshow(s,[]); title('before noise')
%add noise
SNR = -10;
ss = awgn( s,SNR,'measured' );
subplot(1,2,2),imshow(ss,[]); title('add noise')
g = ss - mean( ss(:) ) ; ;   % imperfect reconstruction
figure;imagesc(abs(g));title('Diffracted field');axis image;
axis off; colormap(hot); colorbar;

g=MyC2V(g(:));
transf=MyAdjointOperatorPropagation(g,E,nx,ny,nz,Phase3D,Pupil);
transf=reshape(MyV2C(transf),nx,ny,nz);
figure;imagesc(plotdatacube(abs(transf)));title('Numerical backpropagation');axis image;drawnow;
axis off; colormap(hot); colorbar;
% calculate the PSNR
Ps = sum(sum(( s - mean(mean( s )) ).^2));%signal power
Pn = sum(sum(( ss-s ).^2));           %noise power
snr1=10*log10(Ps/Pn);
%% Propagation operator (5)
A = @(f_twist) MyForwardOperatorPropagation(f_twist,E,nx,ny,nz,Phase3D,Pupil);  % forward propagation operator
AT = @(g) MyAdjointOperatorPropagation(g,E,nx,ny,nz,Phase3D,Pupil);  % backward propagation operator


%% TwIST algorithm (6)
% twist parameters
tau = 0.001;   
piter = 4;
tolA = 1e-12;
iterations = 500;

Psi = @(f,th) MyTVpsi(f,th,0.05,piter,Nx,Ny,Nz);
Phi = @(f) MyTVphi(f,Nx,Ny,Nz);

[f_reconstruct,dummy,obj_twist,...
    times_twist,dummy,mse_twist]= ...
    TwIST(g,A,tau,...
    'AT', AT, ...
    'Psi', Psi, ...
    'Phi',Phi, ...
    'Initialization',2,...
    'Monotone',1,...
    'StopCriterion',1,...
    'MaxIterA',iterations,...
    'MinIterA',iterations,...
    'ToleranceA',tolA,...
    'Verbose', 1);

f_reconstruct=reshape(MyV2C(f_reconstruct),nx,ny,nz);
figure;imagesc(plotdatacube(abs(f_reconstruct)));title('Compressive reconstruction');axis image;drawnow;
axis off; colormap(hot); colorbar;
% show the reconstruction
figure(),imagesc( abs(f_reconstruct(1:64,1:64,2)) );axis off;axis image;drawnow;

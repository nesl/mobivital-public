close all

dataset = readmatrix('YOUR_PATH/240205_userE_tripod_01_3.csv');
figure()

breath_wave = dataset(:,end-1);
subplot(4,1,1)
plot(breath_wave)
title("Truth-Breath")

heart_wave = dataset(:,end);
subplot(4,1,2)
plot(heart_wave)
title("Truth-Heart")

target_bin = 25;
uwb_wave = dataset(:,target_bin+6+6) + 1j.*dataset(:,target_bin+120+6+6);
subplot(4,1,3)
plot(real(uwb_wave))
title("In-pahse")
subplot(4,1,4)
plot(imag(uwb_wave))
title("Quadrature")

figure()
acc_x = dataset(:,1);
subplot(6,1,1)
plot(acc_x)
title("Acc-x")

acc_y = dataset(:,2);
subplot(6,1,2)
plot(acc_y)
title("Acc-y")

acc_z = dataset(:,3);
subplot(6,1,3)
plot(acc_z)
title("Acc-z")

gyro_x = dataset(:,4);
subplot(6,1,4)
plot(gyro_x)
title("Gyro-x")

gyro_y = dataset(:,5);
subplot(6,1,5)
plot(gyro_y)
title("Gyro-y")

gyro_z = dataset(:,6);
subplot(6,1,6)
plot(gyro_z)
title("Gyro-z")


figure()
real_x = dataset(:,10);
subplot(6,1,1)
plot(real_x)
title("Realsense-x")

real_y = dataset(:,11);
subplot(6,1,2)
plot(real_y)
title("Realsense-y")

real_z = dataset(:,12);
subplot(6,1,3)
plot(real_z)
title("Realsense-z")

real_rou = dataset(:,7);
subplot(6,1,4)
plot(real_rou)
title("Realsense-rou")

real_theta = dataset(:,8);
subplot(6,1,5)
plot(real_theta)
title("Realsense-theta")

real_phi = dataset(:,9);
subplot(6,1,6)
plot(real_phi)
title("Realsense-phi")

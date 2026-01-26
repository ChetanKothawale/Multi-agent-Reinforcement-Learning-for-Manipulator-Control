function jacobian_control_4dof
    l1 = 0.228;
    l3 = 0.22;
    l4 = 0.22;

    % [q1 q2 q3 q4 integral_x integral_y integral_z error_prev_x error_prev_y error_prev_z]
    q0 = [0; pi/2; 0; 0; 0; 0; 0; 0; 0; 0];
    
    xd = [-0.24; -0.196; 0.207];
    
    Kp = 7 * eye(3);
    Ki = 1 * eye(3);
    Kd = 0.008 * eye(3);
    
    tspan = linspace(0, 3, 100);
    dt = tspan(2) - tspan(1);
    
    [t, q_hist] = ode45(@(t, q) robot_dynamics(t, q, xd, Kp, Ki, Kd, l1, l3, l4, dt), tspan, q0);
    
    q = q_hist(:, 1:4);
    
    x_hist = zeros(length(t), 3);
    
    figure;
    hold on;
    grid on;
    xlabel('X-axis (m)');
    ylabel('Y-axis (m)');
    zlabel('Z-axis (m)');
    title('4-DOF Robot Motion with PID Control');
    axis equal;
    view(3);
    
    for i = 1:length(t)
        [x_hist(i, :), X_links, Y_links, Z_links] = forward_kinematics(q(i, :)', l1, l3, l4);
        
        cla;
        plot3(x_hist(:, 1), x_hist(:, 2), x_hist(:, 3), 'k--');
        hold on;
        plot3(X_links, Y_links, Z_links, 'ro-', 'LineWidth', 2, 'MarkerSize', 8);
        scatter3(xd(1), xd(2), xd(3), 100, 'g', 'filled');
        
        pause(0.05);
    end
    
    figure('Position', [100 100 1200 500])
    hold on;
    plot(t, x_hist(:,1), 'r', t, x_hist(:,2), 'g', t, x_hist(:,3), 'b');
    plot(t, xd(1)*ones(size(t)), '--r');
    plot(t, xd(2)*ones(size(t)), '--g');
    plot(t, xd(3)*ones(size(t)), '--b');
    title('End-Effector Position');
    xlabel('Time (s)');
    ylabel('Position (m)');
    legend('X','Y','Z','X_d','Y_d','Z_d');
end
%% Robot Dynamics with PID Control
function dqdt = robot_dynamics(~, q, xd, Kp, Ki, Kd, l1, l3, l4, dt)
    q_joints = q(1:4);
    error_prev = q(8:10);
    x = forward_kinematics(q_joints, l1, l3, l4);
    e = xd - x;
    integral = q(5:7) + e*dt;
    
    de_dt = (e - error_prev)/dt;
    J = compute_jacobian(q_joints, l1, l3, l4);
    
    % PID Control law
    u = Kp * e + Ki * integral + Kd * de_dt;
    
    % 1) Inverse of Jacobian
    dq = pinv(J) * u;

    % % 2) Singularity Avoidance using Damped Least Squares (DLS)
    % lambda = 0.1; % Damping factor
    % J_damped = J' * inv(J * J' + lambda^2 * eye(3));
    % dq = J_damped * u;

    max_integral = 10;
    integral = sign(integral) .* min(abs(integral), max_integral);
    prev_integral = integral;
    
    dqdt = [dq; prev_integral; e];
end
%% Forward Kinematics Calculation
function [x, X_links, Y_links, Z_links] = forward_kinematics(q, l1, l3, l4)
    q1 = q(1); q2 = q(2); q3 = q(3); q4 = q(4);

    base = [0; 0; 0]; 
    joint1 = base + [0; 0; l1];
    joint2 = joint1 + [cos(q1) * cos(q2) * l3; sin(q1) * cos(q2) * l3; sin(q2) * l3];
    joint3 = joint2 + [cos(q1) * cos(q2 + q3) * l4; sin(q1) * cos(q2 + q3) * l4; sin(q2 + q3) * l4];

    end_effector = joint3 ;

    X_links = [base(1), joint1(1), joint2(1), joint3(1), end_effector(1)];
    Y_links = [base(2), joint1(2), joint2(2), joint3(2), end_effector(2)];
    Z_links = [base(3), joint1(3), joint2(3), joint3(3), end_effector(3)];

    x = end_effector;
end
%% Computing Jacobian using approximation
function J = compute_jacobian(q, l1, l3, l4)
    delta = 1e-6;
    J = zeros(3,4);
    x0 = forward_kinematics(q, l1, l3, l4);

    for i = 1:4
        q_perturbed = q;
        q_perturbed(i) = q_perturbed(i) + delta;
        x_perturbed = forward_kinematics(q_perturbed, l1, l3, l4);
        J(:, i) = (x_perturbed - x0) / delta;
    end
end
function marl_control_4dof
    
    l1 = 0.228; l3 = 0.22; l4 = 0.22;
    xd = [-0.24; -0.196; 0.207]; 
    
    % MARL hyperparameters
    alpha_a = 0.1;              % Actor learning rate
    alpha_c = 0.2;              % Critic learning rate
    gamma = 0.90;               % Discount factor
    epsilon = 0.2;              % Initial exploration rate
    epsilon_decay = 0.9999;     % Decay rate for exploration
    epsilon_min = 0.01;         % Minimum exploration rate
    num_episodes = 1000;        % Number of training episodes
    steps_per_episode = 100;    % Max steps per episode
    
    actions = [-0.1, -0.02, 0, 0.02, 0.1];
    num_actions = length(actions);
       
    error_range = 0.3; 
    bin_size = 0.05;  
    num_bins = round(2 * error_range / bin_size) + 1; 
     
    Q = cell(4, 1); 
    for i = 1:4
        Q{i} = zeros(num_bins, num_bins, num_bins, num_actions);
    end
    V = zeros(num_bins, num_bins, num_bins); 
    
    episode_rewards = zeros(num_episodes, 1);
    final_errors = zeros(num_episodes, 1);
    
    % Training loop
    for episode = 1:num_episodes
        q = [0; pi/2; 0; 0]; % Initial joint angles
        total_reward = 0;
        
        for step = 1:steps_per_episode
         
            [x, ~, ~, ~] = forward_kinematics(q, l1, l3, l4);
            e = xd - x;
            state = discretize_state(e, error_range, bin_size, num_bins);
             
            delta_q = zeros(4, 1);
            action_indices = zeros(4, 1);
            for agent = 1:4
                if rand < epsilon
                    action_idx = randi(num_actions); 
                else
                    [~, action_idx] = max(Q{agent}(state(1), state(2), state(3), :)); % Exploit
                end
                delta_q(agent) = actions(action_idx);
                action_indices(agent) = action_idx;
            end
            
            q_new = q + delta_q;
            [x_new, ~, ~, ~] = forward_kinematics(q_new, l1, l3, l4);
            e_new = xd - x_new;
            state_new = discretize_state(e_new, error_range, bin_size, num_bins);
            
            position_error = -norm(e_new)^2; 
            movement_penalty = -0.1 * norm(delta_q)^2; 
            reward = position_error + movement_penalty;
            
            % Update critic (V)
            td_error = reward + gamma * V(state_new(1), state_new(2), state_new(3)) ...
                       - V(state(1), state(2), state(3));
            V(state(1), state(2), state(3)) = V(state(1), state(2), state(3)) + alpha_c * td_error;
            
        
            for agent = 1:4
                Q{agent}(state(1), state(2), state(3), action_indices(agent)) = ...
                    Q{agent}(state(1), state(2), state(3), action_indices(agent)) + alpha_a * td_error;
            end
            
            if norm(e_new) < 0.01 
                break;
            end
            
            q = q_new;
            total_reward = total_reward + reward;
        end
        
        % Decay exploration rate
        epsilon = max(epsilon_min, epsilon * epsilon_decay);
        
        % Store episode statistics
        episode_rewards(episode) = total_reward;
        [x_final, ~, ~, ~] = forward_kinematics(q, l1, l3, l4);
        final_errors(episode) = norm(xd - x_final);
        
        % Progress update
        if mod(episode, 50) == 0
            fprintf('Episode %d, Total Reward: %.2f, Final Error: %.4f, Epsilon: %.4f\n', ...
                    episode, total_reward, final_errors(episode), epsilon);
        end
    end
    % Plot training progress
    figure(3);
    subplot(2,1,1);
    plot(episode_rewards);
    title('Episode Rewards during Training');
    xlabel('Episode');
    ylabel('Total Reward');
    grid on;
    
    subplot(2,1,2);
    plot(final_errors);
    title('Final Position Error during Training');
    xlabel('Episode');
    ylabel('Error (m)');
    grid on;
    
    % Test the learned policy
    test_learned_policy(Q, actions, l1, l3, l4, xd, error_range, bin_size, num_bins);
end
%%
function state = discretize_state(e, error_range, bin_size, num_bins)
    e_clamped = max(min(e, error_range), -error_range);
    bins = floor((e_clamped + error_range) / bin_size) + 1;
    bins = max(min(bins, num_bins), 1);
    state = bins;     
end
%%
function [x, X_links, Y_links, Z_links] = forward_kinematics(q, l1, l3, l4)
    q1 = q(1); q2 = q(2); q3 = q(3); q4 = q(4);
    
    c1 = cos(q1); s1 = sin(q1);
    c2 = cos(q2); s2 = sin(q2);
    c3 = cos(q3); s3 = sin(q3);
    c4 = cos(q4); s4 = sin(q4);
    
    X = c1 * (c2 * c3 * (l3 + l4 * c4) - l4 * s2 * s4) - (l3 + l4 * c4) * s1 * s3;
    Y = c2 * c3 * (l3 + l4 * c4) * s1 - l4 * s1 * s2 * s4 + c1 * (l3 + l4 * c4) * s3;
    Z = l1 + c3 * (l3 + l4 * c4) * s2 + l4 * c2 * s4;
    
    x = [X; Y; Z];
    
    % Link coordinates for plotting (approximate)
    link1_end = [0; 0; l1];
    link2_end = [l3*c1*c2; l3*s1*c2; l1+l3*s2];
    link3_end = [X; Y; Z];
    
    X_links = [0, link1_end(1), link2_end(1), link3_end(1)];
    Y_links = [0, link1_end(2), link2_end(2), link3_end(2)];
    Z_links = [0, link1_end(3), link2_end(3), link3_end(3)];
end
%%
function test_learned_policy(Q, actions, l1, l3, l4, xd, error_range, bin_size, num_bins)
    q = [0; pi/2; 0; 0];
    tspan = linspace(0, 3, 100);
    x_hist = zeros(length(tspan), 3);
    
    figure(1);
    hold on;
    grid on;
    title('MARL Trained Policy Trajectory');
    xlabel('X-axis (m)');
    ylabel('Y-axis (m)');
    zlabel('Z-axis (m)');
    axis equal;
    view(3);
    scatter3(xd(1), xd(2), xd(3), 100, 'g', 'filled', 'DisplayName', 'Target');
    
    [~, X_links, Y_links, Z_links] = forward_kinematics(q, l1, l3, l4);
    robot_plot = plot3(X_links, Y_links, Z_links, 'ro-', 'LineWidth', 2, 'MarkerSize', 8);
    path_plot = plot3(NaN, NaN, NaN, 'k--', 'DisplayName', 'Trajectory');
    legend('Location', 'best');
    
    for t = 1:length(tspan)
        [x, X_links, Y_links, Z_links] = forward_kinematics(q, l1, l3, l4);
        x_hist(t, :) = x';
        
        e = xd - x;
        state = discretize_state(e, error_range, bin_size, num_bins);
        
        delta_q = zeros(4, 1);
        for agent = 1:4
            [~, action_idx] = max(Q{agent}(state(1), state(2), state(3), :));
            delta_q(agent) = actions(action_idx);
        end
        q = q + delta_q;
        
        set(robot_plot, 'XData', X_links, 'YData', Y_links, 'ZData', Z_links);
        set(path_plot, 'XData', x_hist(1:t,1), 'YData', x_hist(1:t,2), 'ZData', x_hist(1:t,3));
        drawnow;
        pause(0.01);
    end
    
    figure(2)
    hold on;
    plot(tspan, x_hist(:,1), 'r', 'DisplayName', 'X');
    plot(tspan, x_hist(:,2), 'g', 'DisplayName', 'Y');
    plot(tspan, x_hist(:,3), 'b', 'DisplayName', 'Z');
    plot(tspan, xd(1)*ones(size(tspan)), '--r', 'DisplayName', 'X_d');
    plot(tspan, xd(2)*ones(size(tspan)), '--g', 'DisplayName', 'Y_d');
    plot(tspan, xd(3)*ones(size(tspan)), '--b', 'DisplayName', 'Z_d');
    title('End-Effector Position vs Time');
    xlabel('Time (s)');
    ylabel('Position (m)');
    legend('Location', 'best');
    grid on;
    
    final_error = norm(xd - x_hist(end,:)');
    fprintf('Final position error: %.4f meters\n', final_error);
end
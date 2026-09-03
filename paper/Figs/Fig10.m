% Read the CSV file as a numeric matrix
data = readmatrix('decisionstrategy_3_smoothing01.csv');
heatmap_data = data; % 100x100 matrix

% Define grid parameters
theta_s = linspace(0, 2*pi, size(heatmap_data,1)); % X-axis
theta_m = linspace(0, 2*pi, size(heatmap_data,2)); % Y-axis

% Create figure
figure1 = figure;

% Create axes
axes1 = axes('Parent', figure1);
hold(axes1,'on');

% Plot heatmap
imagesc(theta_s, theta_m, heatmap_data, 'Parent', axes1);
axis xy; % Y increases upwards
colormap(parula);

% Labels
xlabel('Source direction $\theta_s$', 'FontSize', 15, 'Interpreter','latex');
ylabel('Movement direction $\theta_m$', 'FontSize', 15, 'Interpreter','latex');

% Axes properties (tick labels at 0, pi, 2pi)
set(axes1, ...
    'FontSize', 15, ...
    'Layer', 'top', ...
    'TickLabelInterpreter','latex', ...
    'XTick', [0 pi 2*pi], ...
    'XTickLabel', {'0','$\pi$','$2\pi$'}, ...
    'YTick', [0 pi 2*pi], ...
    'YTickLabel', {'0','$\pi$','$2\pi$'});

% Add colorbar with LaTeX labels
c = colorbar(axes1);
c.TickLabels = {'0','0.01','0.015'};
c.TickLabelInterpreter = 'latex';
c.FontSize = 15;
ylabel(c, 'Decision strategy $P_{\Theta_m\mid \Theta_s}(\theta_m\mid \theta_s)$', ...
    'Interpreter','latex', 'FontSize', 15, 'Position', [3.091364227989386,0.010011638454104,0]);

box(axes1,'on');
hold(axes1,'off');

exportgraphics(gcf,'Fig10.eps',...   % since R2020a
    'ContentType','vector',...
    'BackgroundColor','none')
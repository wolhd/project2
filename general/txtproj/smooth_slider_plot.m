function live_drag_dynamic_legend()
    % 1. Specify how many lines you want to test with (Try changing 5 to 3 or 8!)
    numLines = 5;
    
    % Fetch automatically generated random time/trajectory tracks
    [t_cells, x_cells, y_cells] = generate_flexible_data(numLines);

    % Find global absolute time limits
    minTime = Inf; maxTime = -Inf;
    for i = 1:numLines
        minTime = min(minTime, t_cells{i}(1));
        maxTime = max(maxTime, t_cells{i}(end));
    end

    % 2. Setup Figure Canvas
    fig = figure('Name', 'Dynamic Legend Tracker', 'Position', [100 100 800 520]);
    
    % 3. Plot Bounds (Shifted right margin to 0.76 to safely prevent legend overlap)
    ax = axes('Parent', fig, 'Units', 'normalized', 'Position', [0.10, 0.14, 0.66, 0.80]);
    grid(ax, 'on'); hold(ax, 'on');
    
    setappdata(fig, 'timeWindow', 2.0); 
    
    % Get the figure's default color cycle palette
    colorPalette = ax.ColorOrder;
    numColors = size(colorPalette, 1);
    
    % 4. Initialize Plot Object Handles Dynamically Using Standard 'gobjects'
    plotHandles = gobjects(numLines, 1); 
    
    for i = 1:numLines
        % Cycle through standard theme colors if numLines exceeds default palette size
        colorIdx = mod(i-1, numColors) + 1;
        activeColor = colorPalette(colorIdx, :);
        
        % Alternating styles: Every odd track gets connected lines, even tracks get loose points
        if mod(i, 2) ~= 0
            plotHandles(i) = plot(ax, NaN, NaN, '-o', ...
                'Color', activeColor, ...
                'MarkerFaceColor', activeColor, ...
                'MarkerSize', 4, ...
                'LineWidth', 1.2, ...
                'DisplayName', sprintf('Track %d (Line)', i));
        else
            plotHandles(i) = plot(ax, NaN, NaN, 'd', ...
                'Color', activeColor, ...
                'MarkerFaceColor', activeColor, ...
                'MarkerSize', 5, ...
                'DisplayName', sprintf('Track %d (Points)', i));
        end
    end
    hold(ax, 'off');
    
    % Attach the Legend onto the right-hand margin outside the grid canvas
    legend(ax, 'Location', 'eastoutside', 'Box', 'on');
    
    axis(ax, 'equal'); xlim(ax, [-6, 6]); ylim(ax, [-6, 6]);
    xlabel(ax, 'X Position'); ylabel(ax, 'Y Position');

    % 5. Create Compact Slider 
    sld = uicontrol('Parent', fig, 'Style', 'slider', ...
        'Units', 'normalized', ...
        'Position', [0.10, 0.03, 0.55, 0.025], ...
        'min', minTime, ...
        'max', maxTime - getappdata(fig, 'timeWindow'), ...
        'Value', minTime);

    % 6. Create Window Size UI Label & Edit Box on the SAME row
    uicontrol('Parent', fig, 'Style', 'text', ...
        'Units', 'normalized', ...
        'Position', [0.66, 0.025, 0.05, 0.03], ...
        'String', 'Win (s):', ...
        'HorizontalAlignment', 'right', ...
        'BackgroundColor', fig.Color);
    
    txtBox = uicontrol('Parent', fig, 'Style', 'edit', ...
        'Units', 'normalized', ...
        'Position', [0.715, 0.03, 0.05, 0.03], ...
        'String', '2.0', ...
        'HorizontalAlignment', 'center', ...
        'Callback', @(src, ev) updateWindowSize(src, sld, maxTime, minTime, plotHandles, t_cells, x_cells, y_cells, ax));

    % 7. Continuous real-time listener (R2023a syntax)
    sld.addlistener('ContinuousValueChange', ...
        @(src, ev) dragTimeCallback(src, plotHandles, t_cells, x_cells, y_cells, ax));
    
    % Paint initial layout state
    dragTimeCallback(sld, plotHandles, t_cells, x_cells, y_cells, ax);
end

% --- CALLBACK: Slider Dragging ---
function dragTimeCallback(sld, plotHandles, t_cells, x_cells, y_cells, ax)
    fig = sld.Parent;
    timeWindow = getappdata(fig, 'timeWindow');
    
    startTime = sld.Value;
    endTime = startTime + timeWindow;
    
    for i = 1:length(plotHandles)
        t_data = t_cells{i};
        idx = (t_data >= startTime) & (t_data <= endTime);
        
        plotHandles(i).XData = x_cells{i}(idx); 
        plotHandles(i).YData = y_cells{i}(idx);
    end
    
    title(ax, sprintf('Time Window: %.2fs to %.2fs', startTime, endTime));
    drawnow limitrate;
end

% --- CALLBACK: Edit Box Value Changed ---
function updateWindowSize(txtBox, sld, maxTime, minTime, plotHandles, t_cells, x_cells, y_cells, ax)
    fig = txtBox.Parent;
    newVal = str2double(txtBox.String);
    
    if isnan(newVal) || newVal <= 0 || newVal > (maxTime - minTime)
        txtBox.String = num2str(getappdata(fig, 'timeWindow')); 
        return;
    end
    
    setappdata(fig, 'timeWindow', newVal);
    sld.Max = maxTime - newVal;
    
    if sld.Value > sld.Max
        sld.Value = sld.Max;
    end
    
    dragTimeCallback(sld, plotHandles, t_cells, x_cells, y_cells, ax);
end

% --- DYNAMIC CELL ARRAY SIMULATOR ---
function [t_cells, x_cells, y_cells] = generate_flexible_data(numLines)
    t_cells = cell(numLines, 1);
    x_cells = cell(numLines, 1);
    y_cells = cell(numLines, 1);
    
    for i = 1:numLines
        % Randomize individual trace timelines and point densities smoothly
        numPoints = randi([1200, 2500]);
        t_cells{i} = sort(rand(1, numPoints) * 20);
        
        % Generate unique geometric trajectories based on line index
        frequencyMultiplier = 0.3 + (i * 0.2);
        radiusPattern = (t_cells{i}/5) * (1 / (1 + 0.2*i));
        
        x_cells{i} = (radiusPattern + i*0.4) .* sin(t_cells{i} * frequencyMultiplier);
        y_cells{i} = (radiusPattern + i*0.4) .* cos(t_cells{i} * frequencyMultiplier);
    end
end

function live_drag_structure_array()
    % 1. Specify how many lines you want to test with
    numLines = 5;
    
    % Fetch the data bundled neatly into a structure array
    dataStruct = generate_structure_data(numLines);

    % Find global absolute time limits using structure dot-notation
    minTime = Inf; maxTime = -Inf;
    for i = 1:numLines
        minTime = min(minTime, dataStruct(i).time(1));
        maxTime = max(maxTime, dataStruct(i).time(end));
    end

    % 2. Setup Figure Canvas
    fig = figure('Name', 'Structure Array Live Tracker', 'Position', [100 100 800 520]);
    
    % 3. Plot Bounds (Leaves room for the legend on the right)
    ax = axes('Parent', fig, 'Units', 'normalized', 'Position', [0.10, 0.14, 0.66, 0.80]);
    grid(ax, 'on'); hold(ax, 'on');
    
    setappdata(fig, 'timeWindow', 2.0); 
    
    colorPalette = ax.ColorOrder;
    numColors = size(colorPalette, 1);
    
    % 4. Initialize Plot Object Handles Using Standard 'gobjects'
    plotHandles = gobjects(numLines, 1); 
    
    for i = 1:numLines
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
    
    % Attach the Legend
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
        'Callback', @(src, ev) updateWindowSize(src, sld, maxTime, minTime, plotHandles, dataStruct, ax));

    % 7. Continuous real-time listener (R2023a syntax)
    sld.addlistener('ContinuousValueChange', ...
        @(src, ev) dragTimeCallback(src, plotHandles, dataStruct, ax));
    
    % Paint initial layout state
    dragTimeCallback(sld, plotHandles, dataStruct, ax);
end

% --- CALLBACK: Slider Dragging ---
function dragTimeCallback(sld, plotHandles, dataStruct, ax)
    fig = sld.Parent;
    timeWindow = getappdata(fig, 'timeWindow');
    
    startTime = sld.Value;
    endTime = startTime + timeWindow;
    
    % Loop through the structure array using standard dot fields
    for i = 1:length(plotHandles)
        t_data = dataStruct(i).time;
        idx = (t_data >= startTime) & (t_data <= endTime);
        
        plotHandles(i).XData = dataStruct(i).x(idx); 
        plotHandles(i).YData = dataStruct(i).y(idx);
    end
    
    title(ax, sprintf('Time Window: %.2fs to %.2fs', startTime, endTime));
    drawnow limitrate;
end

% --- CALLBACK: Edit Box Value Changed ---
function updateWindowSize(txtBox, sld, maxTime, minTime, plotHandles, dataStruct, ax)
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
    
    dragTimeCallback(sld, plotHandles, dataStruct, ax);
end

% --- DYNAMIC STRUCTURE ARRAY GENERATOR ---
function dataStruct = generate_structure_data(numLines)
    % Initialize an empty structure array with explicit fields
    dataStruct = struct('time', {}, 'x', {}, 'y', {});
    
    for i = 1:numLines
        numPoints = randi([1200, 2500]);
        
        % Populate indices using structure dot notation
        dataStruct(i).time = sort(rand(1, numPoints) * 20);
        
        frequencyMultiplier = 0.3 + (i * 0.2);
        radiusPattern = (dataStruct(i).time / 5) * (1 / (1 + 0.2*i));
        
        dataStruct(i).x = (radiusPattern + i*0.4) .* sin(dataStruct(i).time * frequencyMultiplier);
        dataStruct(i).y = (radiusPattern + i*0.4) .* cos(dataStruct(i).time * frequencyMultiplier);
    end
end

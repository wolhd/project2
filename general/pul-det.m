%% ========================================================================
%  Wideband Radar Pulse Detector — TOA / PW / Center Frequency Extraction
%  No prior knowledge of the emitter beyond the frequency range covered
%  by the receiver's instantaneous bandwidth.
% ========================================================================

clear; clc; close all;

%% ---------------------- USER PARAMETERS --------------------------------
Fs          = 100e6;     % Sample rate of your IQ capture (Hz)
useSimData  = true;      % true = generate test pulses, false = load your own IQ

% STFT parameters (tune these to your shortest expected pulse width)
winLen      = 256;       % samples per FFT window
overlap     = 0.75;      % fractional overlap (0.75 = 75%)
nfft        = 512;       % FFT size (zero-padded for freq interpolation)

% CFAR parameters
guardCells  = 2;         % guard cells around cell-under-test (per side)
trainCells  = 10;        % training/reference cells (per side)
pfa         = 1e-4;      % desired probability of false alarm
minPulseSamples = 3;     % minimum consecutive time-bins to count as a pulse

%% ---------------------- LOAD OR SIMULATE IQ DATA -----------------------
if useSimData
    [iq, t] = generateTestSignal(Fs);
else
    % Replace with your own load, e.g.:
    % iq = readIQFile('capture.bin', Fs);
    error('Set useSimData = true, or provide your own IQ load code here.');
end

%% ---------------------- STFT / SPECTROGRAM ------------------------------
noverlap = round(winLen * overlap);
window   = hamming(winLen);

[S, F, T] = spectrogram(iq, window, noverlap, nfft, Fs, 'centered');
Pxx = abs(S).^2;                 % power spectrogram (freq bins x time bins)
Pxx_dB = 10*log10(Pxx + eps);

fprintf('Spectrogram size: %d freq bins x %d time bins\n', size(Pxx,1), size(Pxx,2));

%% ---------------------- CFAR DETECTION (per frequency row) -------------
% CA-CFAR applied along the time axis, independently for each frequency bin.
% This finds time-localized energy bursts at each frequency -> handles
% unknown carrier frequency naturally.

[nFreq, nTime] = size(Pxx);
detMask = false(nFreq, nTime);

alpha = trainCells*2 * (pfa^(-1/(trainCells*2)) - 1);  % CA-CFAR threshold factor

for fRow = 1:nFreq
    powRow = Pxx(fRow, :);
    for k = (trainCells+guardCells+1):(nTime-trainCells-guardCells)
        leadTrain = powRow(k-trainCells-guardCells : k-guardCells-1);
        lagTrain  = powRow(k+guardCells+1 : k+guardCells+trainCells);
        noiseEst  = mean([leadTrain, lagTrain]);
        threshold = alpha * noiseEst;
        if powRow(k) > threshold
            detMask(fRow, k) = true;
        end
    end
end

%% ---------------------- CLUSTER DETECTIONS INTO PULSES ------------------
% Group connected regions in the time-frequency detection mask into
% individual pulse detections (handles pulses spanning multiple freq bins).

CC = bwconncomp(detMask, 8);   % 8-connectivity (needs Image Processing Toolbox)
stats = regionprops(CC, Pxx, 'PixelIdxList');

pdwList = struct('TOA', {}, 'PW', {}, 'FreqCenter', {}, 'Bandwidth', {}, ...
                  'PeakPower_dB', {});

for i = 1:numel(stats)
    idx = stats(i).PixelIdxList;
    [freqIdx, timeIdx] = ind2sub(size(Pxx), idx);

    if numel(unique(timeIdx)) < minPulseSamples
        continue;   % reject spurious single-bin blips
    end

    % --- Time extent -> TOA and PW ---
    tStart = T(min(timeIdx));
    tEnd   = T(max(timeIdx));
    TOA = tStart;
    PW  = tEnd - tStart;

    % --- Frequency centroid, power-weighted ---
    linPower = Pxx(idx);
    freqVals = F(freqIdx);
    freqCenter = sum(freqVals .* linPower) / sum(linPower);
    bandwidth  = max(freqVals) - min(freqVals);

    % --- Peak power for reference ---
    peakPower_dB = 10*log10(max(linPower));

    pdwList(end+1) = struct( ...
        'TOA', TOA, ...
        'PW', PW, ...
        'FreqCenter', freqCenter, ...
        'Bandwidth', bandwidth, ...
        'PeakPower_dB', peakPower_dB); %#ok<SAGROW>
end

% Sort by time of arrival
[~, order] = sort([pdwList.TOA]);
pdwList = pdwList(order);

%% ---------------------- DISPLAY RESULTS ---------------------------------
fprintf('\nDetected %d pulses:\n', numel(pdwList));
fprintf('%-4s %-12s %-12s %-15s %-12s %-10s\n', ...
    '#', 'TOA (us)', 'PW (us)', 'FreqCenter(MHz)', 'BW (MHz)', 'Pk(dB)');
for i = 1:numel(pdwList)
    fprintf('%-4d %-12.3f %-12.3f %-15.4f %-12.4f %-10.1f\n', i, ...
        pdwList(i).TOA*1e6, pdwList(i).PW*1e6, ...
        pdwList(i).FreqCenter/1e6, pdwList(i).Bandwidth/1e6, ...
        pdwList(i).PeakPower_dB);
end

%% ---------------------- PLOT SPECTROGRAM WITH DETECTIONS ----------------
figure;
imagesc(T*1e6, F/1e6, Pxx_dB);
axis xy;
xlabel('Time (\mus)');
ylabel('Frequency (MHz)');
title('Spectrogram with Detected Pulses');
colorbar;
hold on;
for i = 1:numel(pdwList)
    rectangle('Position', [pdwList(i).TOA*1e6, ...
                            (pdwList(i).FreqCenter - pdwList(i).Bandwidth/2)/1e6, ...
                            pdwList(i).PW*1e6, ...
                            pdwList(i).Bandwidth/1e6], ...
              'EdgeColor', 'r', 'LineWidth', 1.5);
end
hold off;

%% ========================================================================
%  Helper: generate a test signal with a few pulses + noise, for validation
% ========================================================================
function [iq, t] = generateTestSignal(Fs)
    dur = 50e-6;              % total capture duration
    t = (0:1/Fs:dur-1/Fs)';
    N = length(t);

    noisePower = 1e-3;
    iq = sqrt(noisePower/2) * (randn(N,1) + 1i*randn(N,1));

    % Pulse 1: simple CW pulse
    f1 = 10e6; toa1 = 5e-6; pw1 = 2e-6;
    iq = addPulse(iq, t, f1, toa1, pw1, 5, Fs);

    % Pulse 2: different frequency, later TOA
    f2 = -15e6; toa2 = 15e-6; pw2 = 1e-6;
    iq = addPulse(iq, t, f2, toa2, pw2, 5, Fs);

    % Pulse 3: a third pulse, testing PRI-like spacing
    f3 = 10e6; toa3 = 25e-6; pw3 = 2e-6;
    iq = addPulse(iq, t, f3, toa3, pw3, 5, Fs);
end

function iq = addPulse(iq, t, fc, toa, pw, ampSNR, ~)
    idx = (t >= toa) & (t < toa + pw);
    sig = ampSNR * exp(1i*2*pi*fc*t(idx));
    iq(idx) = iq(idx) + sig;
end

function evaluate

addpath('./MMSESTSA85');
addpath('./segsnr/segsnr');
addpath('./matlab_pesq_wrapper')

root_dir = '../dataset/test';
clean_dir = '../dataset/test/clean';

SNRs = ["-5db", "0db", "5db", "10db", "15db", "20db"];
noise_types = ["babble", "destroyerengine", "factory1", "hfchannel"];

segSNR = zeros(1, length(SNRs));
segSNR_WS = zeros(1, length(SNRs));
segSNR_MMSE = zeros(1, length(SNRs));
pesq = zeros(1, length(SNRs));
pesq_WS = zeros(1, length(SNRs));
pesq_MMSE = zeros(1, length(SNRs));

count = 0;
count_WS = 0;
count_MMSE = 0;

parfor j = 1: length(SNRs)
    snr = SNRs(j)
    for noise_type = noise_types
        noisy_dir = fullfile(root_dir, 'noisy', noise_type, snr);
        files = dir(noisy_dir);
        files = files(~ismember({files.name},{'.','..','SegSNRs.csv'}));
        for i = 3: length(files)
            noisy_fn = fullfile(noisy_dir, files(i).name);
            clean_fn = fullfile(clean_dir, files(i).name);
            [y_clean, fs] = audioread(clean_fn);
            [y_noisy, fs] = audioread(noisy_fn);

            y_WS = wdenoise(y_noisy, 'Wavelet','sym8');
            y_MMSE = MMSESTSA85(y_noisy, fs, 0.13);
            % segSNR(j) = segSNR(j) + segsnr(y_clean, y_noisy, fs);
            % segSNR_WS(j) = segSNR_WS(j) + segsnr(y_clean(1: length(y_WS)), y_WS, fs);
            % segSNR_MMSE(j) = segSNR_MMSE(j) + segsnr(y_clean(1: length(y_MMSE)), y_MMSE, fs);

            pesq_ori = pesqbin(y_clean, y_noisy, fs, 'nb');
            pesq(j) = pesq(j) + pesq_ori;
            if pesq_ori == 0
                count = count + 1;
            end
            pesq_ws = pesqbin(y_clean(1: length(y_WS)), y_WS, fs, 'nb');
            pesq_WS(j) = pesq_WS(j) + pesq_ws;
            if pesq_ws == 0
                count_WS = count_WS + 1;
            end
            pesq_mmse = pesqbin(y_clean(1: length(y_MMSE)), y_MMSE, fs, 'nb');
            pesq_MMSE(j) = pesq_MMSE(j) + pesq_mmse;
            if pesq_mmse == 0
                count_MMSE = count_MMSE + 1;
            end
        end
    end
    % s = sprintf('segSNR: %fdb, WS: %fdb, MMSE: %fdb\n', pesq(j) / 1200, pesq_WS(j) / 1200, ...
    %             pesq_MMSE(j) / 1200);
    % fprintf(s);
    % disp(segSNR(j) / 1200);
end

disp(count);
disp(count_WS);
disp(count_MMSE);
disp(pesq);
disp(pesq_WS);
disp(pesq_MMSE);

end

function evaluate

addpath('./MMSESTSA85');
addpath('./segsnr/segsnr');
addpath('./matlab_pesq_wrapper')

root_dir = '../features/pred';
clean_dir = '../Noise_Addition/timit_128/test';

SNRs = ["-5dB", "0dB", "5dB", "10dB", "15dB", "20dB"];
noise_types = ["babble", "destroyerengine", "alarm", "volvo", ...
               "pink", "white"];

segSNR = zeros(1, length(SNRs));
count = 0;

parfor j = 1: length(SNRs)
    snr = SNRs(j)
    for noise_type = noise_types
        noisy_dir = fullfile(root_dir, noise_type, snr);
        files = dir(noisy_dir);
        files = files(~ismember({files.name},{'.','..','SegSNRs.csv'}));
        for i = 1: length(files)
            noisy_fn = fullfile(noisy_dir, files(i).name);
            clean_fn = fullfile(clean_dir, files(i).name);
            [y_clean, fs] = audioread(clean_fn);
            [y_noisy, fs] = audioread(noisy_fn);

            % diff = length(y_clean) - length(y_noisy);
            % y_clean = y_clean(floor(diff / 2): floor(diff / 2) + length(y_noisy) - 1);

            snr_ori = snr_self(y_clean(1: length(y_noisy)), y_noisy);
            segSNR(j) = segSNR(j) + snr_ori;
            count = count + 1;
        end
    end
end

disp(count)
disp(segSNR);
disp(segSNR / 1200);

end

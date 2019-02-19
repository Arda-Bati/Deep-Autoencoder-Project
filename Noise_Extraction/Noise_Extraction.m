clc
clear all
folder1 = "Noisy_Data\";
folder2 = "Noise_Types\";
folder3 = "Clean\";
noise_types = ["airport\", "babble\", "car\", "exhibition\", "restaurant\", "station\", "street\", "train\"];
SNRs = ["0dB","5dB","10dB","15dB"];
noises = cell(6,1);

fname = "";
cur_file = "";

for noise_type = noise_types
    for SNR = SNRs
        cur_file = strcat(folder1,noise_type,SNR,"\");
        wav_files = dir(fullfile(cur_file,'*.wav'));
        n = numel(wav_files);
        cur_noise = [];
        for j = 1:n
            clean_fname = strcat(folder3, "sp", num2str(j,'%02d'), ".wav");
            [clean_speech,fs] = audioread(clean_fname);
            wav = wav_files(j).name;
            [noisy,fs] = audioread(fullfile(cur_file,wav));
            noise = noisy - clean_speech;
            cur_noise = [cur_noise; noise];
        end
        audiowrite(strcat(folder2,noise_type,SNR,".wav"),cur_noise,fs);
    end
end

% noisy_fname = 'sp01_babble_sn15.wav';
% [noisy,fs] = audioread(noisy_fname);
% 
% noise = noisy - clean;
% filename = 'noise.wav';
% audiowrite(filename,noise,fs);
% 
% player = audioplayer(noisy, Fs);
% play(player);
% stop(player);
% 
% pause(player);
% % resume the playback
% resume(player);
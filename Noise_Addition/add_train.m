%Reading target and noise files
clc
clear
result_f = 'results';
noises_f = 'noises';
%timit_f = fullfile('timit_512','timit_64');
timit_f = fullfile('timit_128','timit');
fs = 8000;

mkdir results
train_folder = fullfile('results','train');
test_folder = fullfile('results','test');
mkdir(train_folder);
mkdir(test_folder);

clean_file_names = noises_f;
noise_files = dir(fullfile(clean_file_names,'*.wav'));
noise_count = numel(noise_files);

SNRs = -5:5:20;

noise_types = cell(1,noise_count);
for n_index = 1:(noise_count)
    noise_types{n_index} = noise_files(n_index).name;
    noise_name = noise_types{n_index};
    noise_name = noise_name(1:(size(noise_name,2)-4));
    mkdir(train_folder, noise_name);
    mkdir(test_folder, noise_name);
    
    for SNR = SNRs
        mkdir(fullfile(train_folder, noise_name,strcat(num2str(SNR),'dB')));
        mkdir(fullfile(test_folder, noise_name,strcat(num2str(SNR),'dB')));
    end
end

data_type = 'train';
clean_file_names = fullfile(timit_f,data_type);
clean_files = dir(fullfile(clean_file_names,'*.wav'));
data_size = size(clean_files,1);

loop = 0;
tic
for n_index = 1:noise_count
    
    noise_type = noise_types{n_index};
    noise_name = noise_type(1:(size(noise_type,2)-4));
    noise = audioread(fullfile(noises_f,noise_type));

    for SNR = -5:5:20
        %loop = loop + 1
        names = cell(data_size,1);
        seg_values = zeros(data_size,1);
        
        for speech_index = 1:data_size
     
            clean_fname = clean_files(speech_index).name;
            [clean_speech,fs] = audioread(fullfile(timit_f,data_type,clean_fname));
        
            [noisy, dontcare] = addnoise(clean_speech, noise, SNR);
            ssnr = segsnr(clean_speech, noisy, fs);
            %snr_this = snr_self(orinigal, noisy);
            %fprintf('SegSNR: %0.2f dB\n', ssnr);
            cur_path = fullfile(result_f,data_type,noise_name,strcat(num2str(SNR),'dB'));
            audiowrite(fullfile(cur_path,clean_fname),noisy,fs);
            
            names{speech_index} = clean_fname;
            seg_values(speech_index) = ssnr;      
        end
        writetable(cell2table([names num2cell(seg_values)]),fullfile(cur_path,'SegSNRs.csv'),'writevariablenames',0)
    end
end
toc

data_type = 'test';
clean_file_names = fullfile(timit_f,data_type);
clean_files = dir(fullfile(clean_file_names,'*.wav'));
data_size = size(clean_files,1);
test_size = 200;

speech_indices = randperm(data_size,test_size);
speech_indices = sort(speech_indices);

loop = 0;
tic
for n_index = 1:noise_count
    
    noise_type = noise_types{n_index};
    noise_name = noise_type(1:(size(noise_type,2)-4));
    noise = audioread(fullfile(noises_f,noise_type));

    for SNR = -5:5:20
        %loop = loop + 1
        names = cell(test_size,1);
        seg_values = zeros(test_size,1);
        
        count = 0;
        for speech_index = speech_indices
            
            count = count + 1;
            clean_fname = clean_files(speech_index).name;
            [clean_speech,fs] = audioread(fullfile(timit_f,data_type,clean_fname));
        
            [noisy, ~] = addnoise(clean_speech, noise, SNR);
            ssnr = segsnr(clean_speech, noisy, fs);
            %snr_this = snr_self(orinigal, noisy);
            %fprintf('SegSNR: %0.2f dB\n', ssnr);
            cur_path = fullfile(result_f,data_type,noise_name,strcat(num2str(SNR),'dB'));
            audiowrite(fullfile(cur_path,clean_fname),noisy,fs);
            
            names{count} = clean_fname;
            seg_values(count) = ssnr;      
        end
        writetable(cell2table([names num2cell(seg_values)]),fullfile(cur_path,'SegSNRs.csv'),'writevariablenames',0)
    end
end
toc

% num2str(j,'%02d')
% 
% [orinigal, fs] = audioread('sp10.wav');
% [noise, fs] = audioread('ssn.wav');
% 
% %Desired SNR Level (dB)
% snr = 0;
% 
% [noisy, noise] = addnoise(orinigal, noise, snr);
% 
% ssnr = seg_values(orinigal, noisy, fs);
% %snr_this = snr_self(orinigal, noisy);
% fprintf('SegSNR: %0.2f dB\n', ssnr);
% 
% player = audioplayer(clean_speech, fs);
% play(player);
% stop(player);

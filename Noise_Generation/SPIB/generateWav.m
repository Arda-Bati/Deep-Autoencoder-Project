function generateWav

mat_list = dir('./mat/');
mat_list([mat_list.isdir]) = [];

for i = 1: length(mat_list)
    in_fn = fullfile(mat_list(i).folder, mat_list(i).name);
    [~, name, ~] = fileparts(mat_list(i).name);
    m = load(in_fn)
    y = cell2mat(struct2cell(m));
    size(y)
    y_d = resample(y, 8000, 19980);
    out_fn = fullfile('./wav', strcat(name, '.wav'));
    audiowrite(out_fn, y_d, 8000);
end

end
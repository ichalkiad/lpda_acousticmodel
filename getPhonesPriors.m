m=memmapfile('/home/yannis/Desktop/KALDI_norm_var/frameLabels2a.txt.bin','format','double');
phones_priors = tabulate(m.Data);
save '/home/yannis/Desktop/KALDI_norm_var/phones_priors.mat' phones_priors -v7.3;
quit

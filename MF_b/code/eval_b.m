function out = eval_b(Bu)
    std_1 = std(Bu,1,2);
%     std_1 = sort(std_1,'descend');
%     std_1 = std_1(1:100);
    out = [mean(std_1),std(std_1)];
end
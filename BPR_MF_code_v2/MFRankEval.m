function out = MFRankEval(rawData,testIdx,U,V,par)
out = nan*ones(1,3);
result = makeEval;
rawData = rawData(testIdx,:);
rawData = [rawData,zeros(size(rawData,1),1)];
rawData(:,4) = sum(U(rawData(:,1),:).*V(rawData(:,2),:),2);
[userSet,p] = numunique(rawData(:,1));
temp_out = nan*ones(length(userSet),3);
for i=1:length(userSet)
    sample = rawData(p{i},:);
    item = sample(:,2);
    rating = sample(:,3);
    pred = sample(:,4);  
    [~,pred_idx] = sort(pred,'descend');
    target = zeros(length(rating),1);
    target(rating>3) = 1;
    if sum(target~=zeros(length(rating),1))==0 || sum(target~=ones(length(rating),1))==0
        target = zeros(length(rating),1);
        for j=1:floor(length(rating)/2)
            target(pred_idx==j)=1;
        end
    end
%     temp_out(i,1) = aucEval(target,pred);
%     temp_out(i,2) = ndcgEval(item(pred_idx),item(rating>3),par.topN);
%     temp_out(i,3) = mrrEval(item(pred_idx),item(rating>3),par.topN);
    temp_out(i,1) = result.auc(target,pred);
    temp_out(i,2) = result.ndcg(item(pred_idx),item(rating>3),par.topN);
    temp_out(i,3) = result.mrr(item(pred_idx),item(rating>3),par.topN);
end
del_idx = isnan(temp_out(:,1));
temp_out(del_idx,:) = [];
out(1) = mean(temp_out(:,1));
out(2) = mean(temp_out(:,2));
out(3) = mean(temp_out(:,3));
end


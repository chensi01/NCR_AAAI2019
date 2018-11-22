function out = BPRRankEval(session,testIdx,U,V,par)
out = nan*ones(length(testIdx),5);
result = makeEval;
for i = 1:length(testIdx)
    sample = session{testIdx(i)};
    u = sample.user;
    correctItems = sort(sample.buy(1,:));
    candItems = [sample.noBuy(1,:),sample.buy(1,:)];
    s = U(u,:)*V(candItems,:)';
    [~,idx] = sort(s,'descend');
    rankedItems = candItems(idx);
    target = zeros(length(candItems),1)';
    target(length(sample.noBuy(1,:))+1:end) = 1;
    out(i,1) = result.auc(target,s);
    out(i,2) = result.ndcg(rankedItems,correctItems,par.topN);
    out(i,3) = result.mrr(rankedItems,correctItems,par.topN);
    out(i,4) = result.map(rankedItems,correctItems,par.topN);
    out(i,5) = result.pr(rankedItems,correctItems,par.topN);

%     out(i,1) = aucEval(target,s);
%     out(i,2) = ndcgEval(rankedItems,correctItems,par.topN);
%     out(i,3) = mrrEval(rankedItems,correctItems,par.topN);
%     out(i,4) = mapEval(rankedItems,correctItems,par.topN);
%     out(i,5) = prEval(rankedItems,correctItems,par.topN);
end
for i=1:5
    idx = isnan(out(:,i));
    out(idx,:) = [];
end
out = mean(out);
end
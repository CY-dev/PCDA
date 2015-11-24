[X,y,wt,A] = gendata_std(1000,2000,200,1,0,123,0.1);
[w1, lambda] = mex_PCDA(X,y,3,3,0.1,100,1,1e-5,100,1);
[w2, lambda] = mex_PCDA(X,y,3,3,0.1,100,1,1e-5,100,2);
[w4, lambda] = mex_PCDA(X,y,3,3,0.1,100,1,1e-5,100,4);
[w8, lambda] = mex_PCDA(X,y,3,3,0.1,100,1,1e-5,100,8); % fail
range(w1-w4)

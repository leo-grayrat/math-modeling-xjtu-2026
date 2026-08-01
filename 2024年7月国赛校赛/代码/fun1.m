function result=fun1(loc,sita,l,inc_k,inc_n)
x=(tan(sita)*loc+inc_n)/(tan(sita)-inc_k);
if x <= loc+l/2*cos(sita) && x>=loc-l/2*cos(sita)
    result=1;
else
    result=0;
end
end
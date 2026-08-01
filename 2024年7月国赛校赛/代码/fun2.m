function [count_,light_num]=fun2(x,sita,l,x2_,sita2_)
a=37.5;b=50;c=62.5;
    light_num=1000*cos(sita);
    light_num=fix(light_num);   
    light=ones(light_num,1);
    inc_k=zeros(light_num);%入射斜率
    inc_n=zeros(light_num);%入射截距
    rec_x0=zeros(light_num);
    rec_y0=zeros(light_num);
    rec_k=zeros(light_num);
    dropp=zeros(light_num);
    ta=zeros(light_num);%切线斜率
    tn=zeros(light_num);%切线对应的法线
    light_loc=zeros(light_num,1);
    for j=1:light_num%每面镜子上有light_num束光
        light_loc(1)= x-l/2*cos(sita);
        inc_k(j)=-125/x;%第一次反射光方向
        inc_n(j)=125+(light_loc(j)-x)/(cos(sita)*sin(sita));
        x1=(-2*inc_k(j)*b^2*(inc_n(j)-c)+(4*inc_k(j)^2*b^4*(inc_n(j)-c)^2-4*(inc_k(j)^2*b^2-a^2)*b^2*((inc_n(j)-c)^2-a^2))^(1/2))/(inc_k(j)^2*b^2-a^2)/2;
        x2=(-2*inc_k(j)*b^2*(inc_n(j)-c)-(4*inc_k(j)^2*b^4*(inc_n(j)-c)^2-4*(inc_k(j)^2*b^2-a^2)*b^2*((inc_n(j)-c)^2-a^2))^(1/2))/(inc_k(j)^2*b^2-a^2)/2;
        if x1>0 && x1<30
            rec_x0(j)=x1;
        elseif x2>0 && x2<30
            rec_x0(j)=x2;
        end
        rec_y0(j)=inc_k(j)*rec_x0(j)+inc_n(j);
        ta(j)=a^2*rec_x0(j)/(b^2*(rec_y0(j)-c));
        tn(j)=-1/ta(j);
        rec_k(j)=(2*tn(j)+inc_k(j)*tn(j)^2-inc_k(j))/(1+2*inc_k(j)*tn(j)-tn(j)^2);
        dropp(j)=rec_x0(j)-rec_y0(j)/rec_k(j);
        %EF遮挡
        if light_loc(j)<30
            light(j)=0;
        end
        %相互遮挡
        if fun1(x2_,sita2_,l,inc_k(j),inc_n(j))
           light(j)=0;
        end
        %没有汇聚在EF上
        if rec_x0(j)<-30 || rec_x0(j)>30
            light(j)=0;
        end
        %CD之外
        if dropp(j)>5
            light(j)=0;
        end
        light_loc(j+1)= light_loc(j)+l*cos(sita)/light_num;
    end
    count_=sum(light);
end
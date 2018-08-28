%input a vector of values which sums to 1, where each element is the
%probability of the corresponding event occuring
%output the index of a event sampled from the given probability
%distribution

function index = sampleFromVector(x)

    N = length(x);
    y = x;
    for ii = 1:N
        y(ii) = sum(x(1:ii));
    end
    
    p = rand;
    if p < y(1)
        index = 1;
        return
    end
    for ii = 2:N
        if p >= y(ii-1) && p < y(ii)
            index = ii;
            return
        end
    end
    index = 0;
    
end

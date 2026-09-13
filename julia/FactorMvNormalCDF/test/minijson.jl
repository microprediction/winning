# minimal JSON reader shared with parity/check.jl
mutable struct P
    s::String
    i::Int
end
peek(p) = p.s[p.i]
function skipws(p)
    while p.i <= lastindex(p.s) && isspace(p.s[p.i])
        p.i += 1
    end
end
function jval(p)
    skipws(p)
    c = peek(p)
    c == '{' && return jobj(p)
    c == '[' && return jarr(p)
    c == '"' && return jstr(p)
    if startswith(SubString(p.s, p.i), "true")
        p.i += 4; return true
    elseif startswith(SubString(p.s, p.i), "false")
        p.i += 5; return false
    elseif startswith(SubString(p.s, p.i), "null")
        p.i += 4; return nothing
    end
    j = p.i
    while j <= lastindex(p.s) && (p.s[j] in "+-.eE0123456789")
        j += 1
    end
    v = parse(Float64, p.s[p.i:(j - 1)])
    p.i = j
    return v
end
function jstr(p)
    p.i += 1
    out = IOBuffer()
    while peek(p) != '"'
        c = p.s[p.i]
        if c == '\\'
            p.i += 1
            c = p.s[p.i]
        end
        write(out, c)
        p.i += 1
    end
    p.i += 1
    return String(take!(out))
end
function jarr(p)
    p.i += 1
    out = Any[]
    skipws(p)
    if peek(p) == ']'
        p.i += 1
        return out
    end
    while true
        push!(out, jval(p))
        skipws(p)
        if peek(p) == ','
            p.i += 1
        else
            p.i += 1
            return out
        end
    end
end
function jobj(p)
    p.i += 1
    out = Dict{String,Any}()
    skipws(p)
    if peek(p) == '}'
        p.i += 1
        return out
    end
    while true
        skipws(p)
        k = jstr(p)
        skipws(p)
        p.i += 1                       # ':'
        out[k] = jval(p)
        skipws(p)
        if peek(p) == ','
            p.i += 1
        else
            p.i += 1
            return out
        end
    end
end


parse_json(s::String) = jval(P(s, 1))

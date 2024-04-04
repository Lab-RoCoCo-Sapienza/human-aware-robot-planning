from ltlf2dfa.parser.ltlf import LTLfParser


def prefix_to_infix_LTL(formula):
    if formula.startswith("!"):
        # print("negation formula:", formula)
        arg, rest = prefix_to_infix_LTL(formula[2:])
        # print(f"neg: arg:{arg}, rest:{rest}")
        return f"!({arg})", rest
    elif formula.startswith("X"):
        # print("next")
        arg, rest = prefix_to_infix_LTL(formula[2:])
        # print(f"next: arg:{arg}, rest:{rest}")
        return f"X({arg})", rest
    elif formula.startswith("F"):
        # print("eventually")
        arg, rest = prefix_to_infix_LTL(formula[2:])
        # print(f"event: arg:{arg}, rest:{rest}")
        return f"F({arg})", rest
    elif formula.startswith("G"):
        # print("globally")
        arg, rest = prefix_to_infix_LTL(formula[2:])
        # print(f"glob: arg:{arg}, rest:{rest}")
        return f"G ({arg})", rest
    elif formula.startswith("&"):
        # print("conjunction")
        arg1, rest = prefix_to_infix_LTL(formula[2:])
        arg2, rest = prefix_to_infix_LTL(rest)
        # print(f"conjun: arg1:{arg1}, arg2:{arg2}, rest:{rest})")
        return f"(({arg1}) & ({arg2}))", rest
    elif formula.startswith("|"):
        # print("disjunction")
        arg1, rest = prefix_to_infix_LTL(formula[2:])
        arg2, rest = prefix_to_infix_LTL(rest)
        # print(f"disjun: arg1:{arg1}, arg2:{arg2}, rest:{rest})")
        return f"(({arg1}) | ({arg2}))", rest
    elif formula.startswith("i "):
        # print("implication")
        arg1, rest = prefix_to_infix_LTL(formula[2:])
        arg2, rest = prefix_to_infix_LTL(rest)
        # print(f"implic: arg1:{arg1}, arg2:{arg2}, rest:{rest})")

        return f"(({arg1}) -> ({arg2}))", rest
    elif formula.startswith("e "):
        # print("equivalence")
        arg1, rest = prefix_to_infix_LTL(formula[2:])
        arg2, rest = prefix_to_infix_LTL(rest)
        # print(f"equiv: arg1:{arg1}, arg2:{arg2}, rest:{rest})")

        return f"(({arg1}) <-> ({arg2}))", rest

    elif formula.startswith("U"):
        # print("until")
        arg1, rest = prefix_to_infix_LTL(formula[2:])
        arg2, rest = prefix_to_infix_LTL(rest)
        # print(f"until: arg1:{arg1}, arg2:{arg2}, rest:{rest})")

        return f"(({arg1}) U ({arg2}))", rest
    else:
        # print("symbol")
        symbol = formula.split(" ")[0]
        if len(formula) > len(symbol):
            # print("symbol rest: ", formula[3:])
            return symbol, formula[len(symbol) + 1 :]
        else:
            # print("symbol rest: ")
            return symbol, ""


def check_equivalence(phi_1, phi_2):
    equivalence_formula = "(({}) <-> ({}))".format(phi_1, phi_2)

    parser = LTLfParser()
    formula_str = equivalence_formula
    formula = parser(formula_str)
    dfa = formula.to_dfa()
    print(dfa)
    return (
        dfa
        == 'digraph MONA_DFA {\n rankdir = LR;\n center = true;\n size = "7.5,10.5";\n edge [fontname = Courier];\n node [height = .5, width = .5];\n node [shape = doublecircle]; 1;\n node [shape = circle]; 1;\n init [shape = plaintext, label = ""];\n init -> 1;\n 1 -> 1 [label="true"];\n}'
    )

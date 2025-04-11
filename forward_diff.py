import ir
ir.generate_asdl_file()
import _asdl.loma as loma_ir
import irmutator
import autodiff

def forward_diff(diff_func_id : str,
                 structs : dict[str, loma_ir.Struct],
                 funcs : dict[str, loma_ir.func],
                 diff_structs : dict[str, loma_ir.Struct],
                 func : loma_ir.FunctionDef,
                 func_to_fwd : dict[str, str]) -> loma_ir.FunctionDef:
    """ Given a primal loma function func, apply forward differentiation
        and return a function that computes the total derivative of func.

        For example, given the following function:
        def square(x : In[float]) -> float:
            return x * x
        and let diff_func_id = 'd_square', forward_diff() should return
        def d_square(x : In[_dfloat]) -> _dfloat:
            return make__dfloat(x.val * x.val, x.val * x.dval + x.dval * x.val)
        where the class _dfloat is
        class _dfloat:
            val : float
            dval : float
        and the function make__dfloat is
        def make__dfloat(val : In[float], dval : In[float]) -> _dfloat:
            ret : _dfloat
            ret.val = val
            ret.dval = dval
            return ret

        Parameters:
        diff_func_id - the ID of the returned function
        structs - a dictionary that maps the ID of a Struct to 
                the corresponding Struct
        funcs - a dictionary that maps the ID of a function to 
                the corresponding func
        diff_structs - a dictionary that maps the ID of the primal
                Struct to the corresponding differential Struct
                e.g., diff_structs['float'] returns _dfloat
        func - the function to be differentiated
        func_to_fwd - mapping from primal function ID to its forward differentiation
    """

    # HW1 happens here. Modify the following IR mutators to perform
    # forward differentiation.

    # Apply the differentiation.
    class FwdDiffMutator(irmutator.IRMutator):
        def mutate_function_def(self, node):
            # HW1: TODO
            new_args = [loma_ir.Arg(arg.id, autodiff.type_to_diff_type(diff_structs, arg.t), arg.i) for arg in node.args]
            new_body = [self.mutate_stmt(stmt) for stmt in node.body]
            print(new_body)
            return loma_ir.FunctionDef(diff_func_id, new_args, new_body, node.is_simd, autodiff.type_to_diff_type(diff_structs, node.ret_type))

        def mutate_return(self, node):
            # HW1: TODO
            # print(node.val)
            val, dval = self.mutate_expr(node.val)
            # print(val, dval)
            # print("----------------------")
            return loma_ir.Return(loma_ir.Call("make__dfloat",
                [val, dval]
            ))
            # return super().mutate_return(node)

        def mutate_declare(self, node):
            # HW1: TODO
            # print(node)
            # if node.val is not None:
            #     val, dval = self.mutate_expr(node.val)
            #     d_val = loma_ir.Call("make__dfloat",
            #         [val, dval]
            #     )
            print("declare")
            # print(self.mutate_expr(node.val))
            return loma_ir.Declare(
                node.target,
                autodiff.type_to_diff_type(diff_structs, node.t),
                loma_ir.Call("make__dfloat",
                    list(self.mutate_expr(node.val))
                ) if node.val is not None else None
            )
            # return super().mutate_declare(node)

        def mutate_assign(self, node):
            # HW1: TODO
            # print(node)
            return loma_ir.Assign(
                node.target,
                loma_ir.Call(
                    "make__dfloat",
                    list(self.mutate_expr(node.val))
                )
            )
            # return super().mutate_assign(node)

        def mutate_ifelse(self, node):
            # HW3: TODO
            return super().mutate_ifelse(node)

        def mutate_while(self, node):
            # HW3: TODO
            return super().mutate_while(node)

        def mutate_const_float(self, node):
            # print(node)
            # val = loma_ir.StructAccess(node, 'val')
            # print(val)
            return node, loma_ir.ConstFloat(0.0)
            # return loma_ir.Call(
            #     'make__dfloat',
            #     [node, loma_ir.ConstFloat(0.0)]
            # )
            # HW1: TODO
            # return super().mutate_const_float(node)

        def mutate_const_int(self, node):
            # HW1: TODO
            return super().mutate_const_int(node)

        def mutate_var(self, node):
            # HW1: TODO
            val = loma_ir.StructAccess(node, 'val')
            dval = loma_ir.StructAccess(node, 'dval')
            return val, dval
            # return super().mutate_var(node)
            # return loma_ir.Call(
            #     'make__dfloat',
            #     [node, loma_ir.ConstFloat(0.0)]
            # )

        def mutate_array_access(self, node):
            # HW1: TODO
            return super().mutate_array_access(node)

        def mutate_struct_access(self, node):
            # HW1: TODO
            return super().mutate_struct_access(node)

        def mutate_add(self, node):
            # HW1: TODO
            print("add")
            left_val, left_dval = self.mutate_expr(node.left)
            right_val, right_dval = self.mutate_expr(node.right)
            return loma_ir.BinaryOp(
                        loma_ir.Add(),
                        left_val,
                        right_val
                    ), loma_ir.BinaryOp(
                        loma_ir.Add(),
                        left_dval,
                        right_dval
                    )
            # left_val = loma_ir.StructAccess(left, 'val')
            # right_val = loma_ir.StructAccess(right, 'val')
            # left_dval = loma_ir.StructAccess(left, 'dval')
            # right_dval = loma_ir.StructAccess(right, 'dval')
            # return loma_ir.Call(
            #     'make__dfloat',
            #     [
            #         loma_ir.BinaryOp(
            #             loma_ir.Add(),
            #             left_val,
            #             right_val
            #         ),
            #         loma_ir.BinaryOp(
            #             loma_ir.Add(),
            #             left_dval,
            #             right_dval
            #         )
            #     ]
            # )
            # return super().mutate_add(node)

        def mutate_sub(self, node):
            # HW1: TODO
            # return super().mutate_sub(node)
            left_val, left_dval = self.mutate_expr(node.left)
            right_val, right_dval = self.mutate_expr(node.right)
            return loma_ir.BinaryOp(
                        loma_ir.Sub(),
                        left_val,
                        right_val
                    ), loma_ir.BinaryOp(
                        loma_ir.Sub(),
                        left_dval,
                        right_dval
                    )

        def mutate_mul(self, node):
            # HW1: TODO
            # return super().mutate_mul(node)
            left_val, left_dval = self.mutate_expr(node.left)
            right_val, right_dval = self.mutate_expr(node.right)
            left_d = loma_ir.BinaryOp(
                loma_ir.Mul(),
                left_dval,
                right_val
            )
            right_d = loma_ir.BinaryOp(
                loma_ir.Mul(),
                left_val,
                right_dval
            )
            return loma_ir.BinaryOp(
                        loma_ir.Mul(),
                        left_val,
                        right_val
                    ), loma_ir.BinaryOp(
                        loma_ir.Add(),
                        left_d,
                        right_d
                    )

        def mutate_div(self, node):
            # HW1: TODO
            # return super().mutate_div(node)
            left_val, left_dval = self.mutate_expr(node.left)
            right_val, right_dval = self.mutate_expr(node.right)
            up_d = loma_ir.BinaryOp(
                loma_ir.Mul(),
                left_dval,
                right_val
            )
            up_d2 = loma_ir.BinaryOp(
                loma_ir.Mul(),
                left_val,
                right_dval
            )
            up = loma_ir.BinaryOp(
                loma_ir.Sub(),
                up_d,
                up_d2
            )
            down = loma_ir.BinaryOp(
                loma_ir.Mul(),
                right_val,
                right_val
            )
            return loma_ir.BinaryOp(
                        loma_ir.Div(),
                        left_val,
                        right_val
                    ), loma_ir.BinaryOp(
                        loma_ir.Div(),
                        up,
                        down
                    )

        def mutate_call(self, node):
            # HW1: TODO
            print("mutate_call")
            print(node.args)
            id = node.id
            x = node.args[0]
            x_val, x_dval = self.mutate_expr(x)
            if len(node.args) > 1:
                y = node.args[1]
                y_val, y_dval = self.mutate_expr(y)
            if id == 'sin':
                return loma_ir.Call(
                            "sin",
                            [x_val]
                        ), loma_ir.BinaryOp(
                            loma_ir.Mul(),
                            loma_ir.Call(
                                "cos",
                                [x_val]
                            ),
                            x_dval
                        )
            elif id == 'cos':
                return loma_ir.Call(
                            "cos",
                            [x_val]
                        ), loma_ir.BinaryOp(loma_ir.Sub(), 
                            loma_ir.ConstFloat(0.0),
                                loma_ir.BinaryOp(
                                loma_ir.Mul(),
                                loma_ir.Call(
                                    "sin",
                                    [x_val]
                                ),
                                x_dval
                            )
                         )
            elif id == 'sqrt':
                val = loma_ir.Call(
                            "sqrt",
                            [x_val]
                        )
                dval = loma_ir.BinaryOp(
                            loma_ir.Div(),
                            loma_ir.BinaryOp(
                                loma_ir.Mul(),
                                x_dval,
                                loma_ir.ConstFloat(0.5)
                            ),
                            loma_ir.Call(
                                "sqrt",
                                [x_val]
                            )
                        )
                return val, dval
            elif id == 'pow':
                val = loma_ir.Call(
                            "pow",
                            [x_val,
                             y_val]
                        )
                dval_left = loma_ir.BinaryOp(
                    loma_ir.Mul(),
                    x_dval,
                    loma_ir.BinaryOp(
                        loma_ir.Mul(),
                        y_val,
                        loma_ir.Call(
                            "pow",
                            [x_val,
                             loma_ir.BinaryOp(loma_ir.Sub(), y_val, loma_ir.ConstFloat(1.0))]
                        )
                    )
                )
                dval_right = loma_ir.BinaryOp(
                    loma_ir.Mul(),
                    y_dval,
                    loma_ir.BinaryOp(
                        loma_ir.Mul(),
                        loma_ir.Call(
                            "pow",
                            [x_val,
                             y_val]), 
                        loma_ir.Call(
                            "log",
                            [x_val]
                        )
                    )
                )
                return val, loma_ir.BinaryOp(
                    loma_ir.Add(),
                    dval_left,
                    dval_right
                )
            elif id == 'exp':
                val = loma_ir.Call(
                            "exp",
                            [x_val]
                        )
                dval = loma_ir.BinaryOp(
                            loma_ir.Mul(),
                            loma_ir.Call(
                                "exp",
                                [x_val]
                            ),
                            x_dval
                        )
                return val, dval
            elif id == 'log':
                val = loma_ir.Call(
                            "log",
                            [x_val]
                        )
                dval = loma_ir.BinaryOp(
                            loma_ir.Div(),
                            x_dval,
                            x_val
                        )
                return val, dval
            # return super().mutate_call(node)

    return FwdDiffMutator().mutate_function_def(func)

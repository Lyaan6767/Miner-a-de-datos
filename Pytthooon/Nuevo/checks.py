# Para completar el if
if ok:
    print("\n✅ Your output neuron computes  h1 AND NOT h2")
else:
        print("\n❌ Not yet. Hint: a positive weight for h1, a negative weight for h2,"
              " and a bias that keeps the neuron off when h1 = 0.")


_FWD_X = np.array([[0.5, -1.0], [2.0, 0.3], [-1.2, 0.8]])
_FWD_P = [np.array([[0.4, -0.7], [1.2, 0.5]]), np.array([0.1, -0.2]), np.array([[1.5], [-2.0]]), np.array([0.3])]
_FWD_H = np.array([[0.289050497374996, 0.259225100817846], [0.7790261077798122, 0.19000156601531298], [0.6410674063348171, 0.7388500060842489]])
_FWD_Y = np.array([[0.5535749291196774], [0.7481059462133204], [0.4461845376865776]])
_FWD_Z2 = np.array([[0.21512554442680198], [1.0885360296390922], [-0.21609890266627213]])
_BWD_Y = np.array([[1.0], [0.0], [1.0]])
_BWD_G = [np.array([[0.18230690861743198, -0.21042427326421692], [0.014214413278599018, -0.023185921900629725]]), np.array([-0.04519557365334608, 0.05163380175036876]), np.array([[0.03290721089782937], [-0.12758994679349683]]), np.array([-0.08404486232680819])]


def _explain(e):
    msg = str(e).lower()
    if "ellipsis" in msg:
        return "❌ Replace the  ...  placeholders with your code first."
    hint = ""
    if "broadcast" in msg:
        hint = "\n   Hint: use @ for a matrix product (* multiplies element by element)."
    elif "matmul" in msg or "mismatch" in msg or "not aligned" in msg:
        hint = "\n   Hint: the shapes don't line up. Do you need a transpose (.T)?"
    return f"❌ It doesn't run yet -> {type(e).__name__}: {e}{hint}"


def check_forward(forward):
    try:
        H, Y_hat = forward(_FWD_X.copy(), [p.copy() for p in _FWD_P])
    except Exception as e:
        print(_explain(e))
        return
    if H is Ellipsis or Y_hat is Ellipsis:
        print("❌ Replace the  ...  placeholders with your code first.")
        return
    H, Y_hat = np.asarray(H, dtype=float), np.asarray(Y_hat, dtype=float)
    if H.shape != _FWD_H.shape:
        print(f"❌ H has shape {H.shape}, expected {_FWD_H.shape} (one row per flower, one column per hidden neuron).")
    elif not np.allclose(H, _FWD_H):
        print("❌ H has the right shape but the wrong values. "
              "Hidden layer = weighted sum (X @ W1) + bias, passed through the sigmoid.")
    elif Y_hat.shape != _FWD_Y.shape or not np.allclose(Y_hat, _FWD_Y):
        print("❌ H is right, but Y_hat is not. Output layer = weighted sum of H (H @ W2) + bias, "
              "passed through out_act.")
    else:
        try:
            _, Z_out = forward(_FWD_X.copy(), [p.copy() for p in _FWD_P], out_act=lambda z: z)
            uses_out_act = np.allclose(np.asarray(Z_out, dtype=float), _FWD_Z2)
        except Exception:
            uses_out_act = False
        if uses_out_act:
            print("✅ forward() is correct!")
        else:
            print("❌ Almost! At the output, use out_act(...) instead of sigmoid(...), "
                  "so that Part 4 can swap in a different output activation.")


def check_backward(backward):
    try:
        grads = backward(_FWD_X.copy(), _BWD_Y.copy(), _FWD_H.copy(), _FWD_Y.copy(), [p.copy() for p in _FWD_P])
        dW1, db1, dW2, db2 = [np.asarray(g, dtype=float) for g in grads]
    except Exception as e:
        print(_explain(e))
        return
    if not (np.allclose(dW2, _BWD_G[2]) and np.allclose(db2, _BWD_G[3])):
        print("❌ The output-layer lines were given to you: make sure they weren't changed.")
    elif db1.shape != _BWD_G[1].shape or not np.allclose(db1, _BWD_G[1]):
        print("❌ delta1 is not right. It is (delta2 sent back through W2) times the slope of the sigmoid, H * (1 - H). "
              "Use @ with W2.T to send it back, and * for the element-wise product.")
    elif dW1.shape != _BWD_G[0].shape or not np.allclose(dW1, _BWD_G[0]):
        print("❌ delta1 is right, but dW1 is not. Mirror  dW2 = H.T @ delta2  one layer down: "
              "the input of the hidden layer is X.")
    else:
        print("✅ backward() is correct. You just implemented backpropagation!")


print("Helpers loaded ✅")

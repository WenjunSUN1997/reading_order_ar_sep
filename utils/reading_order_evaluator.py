from sklearn.metrics import accuracy_score
import evaluate

def acc_eval(y_true, y_pred):
    return accuracy_score(y_true, y_pred)

def bleu_eval(y_true, y_pred):
    evaluator = evaluate.load("bleu")
    y_pred = " ".join(str(x) for x in y_pred)
    y_true = " ".join(str(x) for x in y_true)
    # predictions = ["hello there general kenobi", "foo bar foobar"]
    # references = [["hello there general kenobi"],["foo bar foobar"]]
    # 使用evauate时必须注明prediction和reference否则会报错
    return evaluator.compute(predictions=[y_pred], references=[[y_true]])['bleu']
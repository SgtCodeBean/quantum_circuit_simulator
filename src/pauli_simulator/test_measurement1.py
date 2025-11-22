from Tableau_Ver2 import Tableau

t = Tableau(2, enable_metrics=True)
t.x(0) 
t.measure(0)
t.measure(1)

m = t.get_metrics()
assert m['measurements']['deterministic'] == 2
assert m['measurements']['probabilistic'] == 0
print("Deterministic test passed")

t2 = Tableau(2, enable_metrics=True)
t2.h(0)
t2.measure(0)

m2 = t2.get_metrics()
assert m2['measurements']['probabilistic'] == 1
assert m2['measurements']['deterministic'] == 0
print("Probabilistic test passed")

t3 = Tableau(1, enable_metrics=True)
for _ in range(100):
    t3.h(0)
    t3.measure(0)

m3 = t3.get_metrics()
print(f"Outcomes over 100 measurements: 0={m3['measurements']['outcomes'][0]}%, 1={m3['measurements']['outcomes'][1]}%")

t3.print_metrics()

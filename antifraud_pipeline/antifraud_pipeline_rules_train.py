# =============================================================================
# 5. Правила первого уровня
# =============================================================================
print("\n" + "=" * 80)
print("РАЗДЕЛ 5. КОМБИНИРОВАННАЯ АРХИТЕКТУРА: ПРАВИЛА + ML")
print("=" * 80)


class RuleBasedFilter:
    def __init__(self):
        self.rules = []
        self.rule_stats = defaultdict(lambda: {"blocked": 0, "passed": 0, "safe": 0})

    def add_rule(self, name, condition_fn, action="block"):
        self.rules.append({"name": name, "condition": condition_fn, "action": action})

    def apply(self, X, return_details=False):
        n = len(X)
        rule_predictions = np.full(n, -1, dtype=int)
        triggered_rules = [[] for _ in range(n)]
        for rule in self.rules:
            try:
                mask = rule["condition"](X)
                if isinstance(mask, pd.Series):
                    mask = mask.values
                if rule["action"] == "block":
                    to_block = mask & (rule_predictions == -1)
                    rule_predictions[to_block] = 1
                    self.rule_stats[rule["name"]]["blocked"] += int(to_block.sum())
                elif rule["action"] == "safe":
                    to_safe = mask & (rule_predictions == -1)
                    rule_predictions[to_safe] = 0
                    self.rule_stats[rule["name"]]["safe"] += int(to_safe.sum())
                for i in np.where(mask)[0]:
                    triggered_rules[i].append(rule["name"])
            except Exception as e:
                print(f"  Ошибка в правиле '{rule['name']}': {e}")
        mask_grey = rule_predictions == -1
        self.rule_stats["__summary__"] = {
            "total": n,
            "blocked_by_rules": int((rule_predictions == 1).sum()),
            "safe_by_rules": int((rule_predictions == 0).sum()),
            "grey_zone": int(mask_grey.sum()),
        }
        if return_details:
            return mask_grey, rule_predictions, triggered_rules
        return mask_grey, rule_predictions

    def print_stats(self):
        summary = self.rule_stats.get("__summary__", {})
        print("\n  --- Статистика правил ---")
        print(f"  Всего транзакций: {summary.get('total', 0)}")
        print(f"  Заблокировано правилами: {summary.get('blocked_by_rules', 0)}")
        print(f"  Безопасно по правилам: {summary.get('safe_by_rules', 0)}")
        print(f"  Серая зона (→ ML): {summary.get('grey_zone', 0)}")
        for name, stats in self.rule_stats.items():
            if name != "__summary__":
                total_triggered = stats["blocked"] + stats["safe"]
                if total_triggered > 0:
                    print(f"    Правило '{name}': blocked={stats['blocked']}, safe={stats['safe']}")


rule_filter = RuleBasedFilter()
# Синхронно с antifraud_stages.stage_05_rules: только эти три правила.
if "puid_orders_1h_without_refunds" in X_train.columns and "order_loan" in X_train.columns:
    orders_1h_p95 = X_train["puid_orders_1h_without_refunds"].quantile(0.95)
    rule_filter.add_rule(
        "extreme_velocity_1h",
        lambda X, tv=orders_1h_p95: (
            (X["puid_orders_1h_without_refunds"] > tv) & (X["order_loan"] > 2000)
        ),
        action="block",
    )
if "glue_size" in X_train.columns:
    gs = float(X_train["glue_size"].quantile(0.995))
    rule_filter.add_rule("glue_size_spike", lambda X, a=gs: X["glue_size"] >= a, action="block")
if "glue_ead" in X_train.columns:
    ge = float(X_train["glue_ead"].quantile(0.995))
    rule_filter.add_rule("glue_ead_spike", lambda X, a=ge: X["glue_ead"] >= a, action="block")

mask_grey_test, rule_preds_test = rule_filter.apply(X_test)
rule_filter.print_stats()
rule_blocked = rule_preds_test == 1
rule_safe = rule_preds_test == 0
if rule_blocked.sum() > 0:
    print(f"\n  Precision блокировок по правилам: {y_test[rule_blocked].mean():.4f}")
if rule_safe.sum() > 0:
    print(f"  Доля фрода среди «безопасных» по правилам: {y_test[rule_safe].mean():.4%}")
print(f"  Доля транзакций, переданных в ML: {mask_grey_test.mean():.2%}")

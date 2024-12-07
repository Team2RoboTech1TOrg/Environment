1. Для запуска на сервере необходимо клонировать репозиторий:
```
git clone https://github.com/Team2RoboTech1TOrg/Environment.git
```

2. Если вы хотите клонировать определенную ветку:
```
git clone -b branch_name https://github.com/Team2RoboTech1TOrg/Environment.git
```

3. Для тестирования без рендера в файле main.py указать:
```
server = True
```

4. Для логирования в csv в файле app.py/app_server.py аргумент log:
```
test = TestingModel(env, model, log=True)
```
По умолчанию стоит 8 миссий.
Логи сохраняются путем добавления в существующие файлы в папку logging_system.

5. В файле policy.py можно настроить политику (читаем документацию)
```
self.net_arch = {"pi": [128, 64, 32], "vf": [128, 64, 32]}
self.activation_fn = th.nn.ReLU
optimizer_class = ...
```

6. ExplorationScenario.py - настройка мин и макс количества шагов в миссии
```
self.max_steps = self.grid_size ** 2 * 10
self.min_steps = self.grid_size ** 2 * 2
```

7. ExplorationScenario.py функция check_agents_distance,
можно поработать с настройкой расстояния между агентами.

8. Попробовать со штрафом, если клетка уже посещена и без в ExplorationScenario.py
```
reward -= c.PENALTY_RETURN
```
9. В ExplorationScenario.py попробовать разные критерии и значения динамики для награды.
```
if known_targets > self.count_targets * 0.9:
    self.reward_coef *= 1.01  # dynamical coefficient
    reward += c.REWARD_EXPLORE * self.reward_coef
elif known_targets > self.count_targets * 0.75:
    self.reward_coef *= 1.001  # dynamical coefficient
    reward += c.REWARD_EXPLORE * self.reward_coef
else:
    reward += c.REWARD_EXPLORE
```

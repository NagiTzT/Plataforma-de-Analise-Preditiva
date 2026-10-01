import unittest

import robo_auto


class RoboMenuTests(unittest.TestCase):
    def test_command_runs_synchronously_before_returning(self):
        events = []

        def fake_radar():
            events.append("inicio")
            events.append("fim")

        original = robo_auto.COMANDOS_INTERATIVOS["3"]
        robo_auto.COMANDOS_INTERATIVOS["3"] = fake_radar
        try:
            keep_running = robo_auto.executar_comando_interativo("3")
        finally:
            robo_auto.COMANDOS_INTERATIVOS["3"] = original
        self.assertTrue(keep_running)
        self.assertEqual(events, ["inicio", "fim"])

    def test_exit_is_clean(self):
        self.assertFalse(robo_auto.executar_comando_interativo("sair"))


if __name__ == "__main__":
    unittest.main()

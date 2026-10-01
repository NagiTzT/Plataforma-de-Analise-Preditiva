import unittest

from ticket_engine import selecionar_grupos_bilhetes


def game(identifier, confidence, league, pick="MANDANTE"):
    return {"ID": str(identifier), "Confiança": confidence, "Liga": league,
            "Liga_Exata": league, "Pick": pick, "Vencedor Escolhido": pick}


class TicketEngineTests(unittest.TestCase):
    def test_selects_one_strict_four_leg_ticket(self):
        jogos = [
            game(1, 55, "A"), game(2, 54, "A"), game(3, 53, "B"),
            game(4, 52, "C"), game(5, 51, "D"), game(6, 60, "E", "EMPATE"),
        ]
        tickets = selecionar_grupos_bilhetes(jogos, min_confianca=42, max_bilhetes=1)
        self.assertEqual(len(tickets), 1)
        self.assertEqual([j["ID"] for j in tickets[0]["Jogos"]], ["1", "3", "4", "5"])
        self.assertEqual(tickets[0]["Categoria"], "ELITE")
        self.assertAlmostEqual(tickets[0]["Probabilidade Conjunta"], .55 * .53 * .52 * .51)

    def test_does_not_relax_unique_leagues(self):
        jogos = [game(i, 60 - i, "A") for i in range(1, 8)]
        self.assertEqual(selecionar_grupos_bilhetes(jogos), [])

    def test_can_relax_leagues_without_repeating_games(self):
        jogos = [game(i, 60 - i, "A") for i in range(1, 9)]
        tickets = selecionar_grupos_bilhetes(jogos, relaxar_ligas=True)
        self.assertEqual(len(tickets), 2)
        ids = [j["ID"] for ticket in tickets for j in ticket["Jogos"]]
        self.assertEqual(len(ids), 8)
        self.assertEqual(len(set(ids)), 8)
        self.assertTrue(all(len(ticket["Jogos"]) == 4 for ticket in tickets))

    def test_open_policy_leaves_only_lowest_confidence_remainder(self):
        jogos = [game(i, 70 - i, "A" if i < 8 else f"L{i}") for i in range(1, 10)]
        tickets = selecionar_grupos_bilhetes(
            jogos, min_confianca=0, permitir_empate=True,
            ligas_unicas=True, relaxar_ligas=True)
        selected = {j["ID"] for ticket in tickets for j in ticket["Jogos"]}
        self.assertEqual(len(tickets), 2)
        self.assertNotIn("9", selected)

    def test_rejects_low_confidence_and_draws(self):
        jogos = [game(1, 41, "A"), game(2, 60, "B", "EMPATE"),
                 game(3, 50, "C"), game(4, 50, "D"), game(5, 50, "E")]
        self.assertEqual(selecionar_grupos_bilhetes(jogos, min_confianca=42), [])

    def test_default_is_50_and_unlimited_complete_groups(self):
        jogos = [game(i, 60, f"L{i}") for i in range(1, 9)]
        jogos.append(game(9, 49, "L9"))
        tickets = selecionar_grupos_bilhetes(jogos)
        self.assertEqual(len(tickets), 2)
        self.assertNotIn("9", {j["ID"] for t in tickets for j in t["Jogos"]})

    def test_low_probability_ticket_is_labeled_high_risk(self):
        jogos = [game(i, 38, f"L{i}") for i in range(1, 5)]
        tickets = selecionar_grupos_bilhetes(
            jogos, min_confianca=0, permitir_empate=True,
            ligas_unicas=True, relaxar_ligas=True,
        )
        self.assertEqual(tickets[0]["Categoria"], "ALTO RISCO")

    def test_reliable_decisive_games_are_grouped_before_uncertain_games(self):
        jogos = [game(i, 50, f"L{i}") for i in range(1, 9)]
        for jogo in jogos[:4]:
            jogo.update({
                "Qualidade_Contexto": .8, "Confiabilidade_Amostra": 1.0,
                "Margem_Probabilidade": .16, "Entropia_Normalizada": .70,
                "Contexto_Detalhado_Ambos": 1.0,
            })
        for jogo in jogos[4:]:
            jogo.update({
                "Qualidade_Contexto": .2, "Confiabilidade_Amostra": .2,
                "Margem_Probabilidade": .01, "Entropia_Normalizada": .99,
            })
        tickets = selecionar_grupos_bilhetes(jogos, max_bilhetes=1)
        self.assertEqual({j["ID"] for j in tickets[0]["Jogos"]}, {"1", "2", "3", "4"})


if __name__ == "__main__":
    unittest.main()

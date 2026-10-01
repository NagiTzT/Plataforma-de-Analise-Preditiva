import concurrent.futures
import os
import tempfile
import unittest

from telegram_delivery import (
    claim_ticket_delivery,
    delivery_fingerprint,
    mark_ticket_failed,
    mark_ticket_sent,
)


class TelegramDeliveryTests(unittest.TestCase):
    def setUp(self):
        handle, self.db_path = tempfile.mkstemp(suffix=".db")
        os.close(handle)

    def tearDown(self):
        if os.path.exists(self.db_path):
            os.remove(self.db_path)

    def test_only_one_concurrent_process_claims_same_composition(self):
        payload_hash = delivery_fingerprint([
            {"ID": "10", "Vencedor Escolhido": "MANDANTE"},
            {"ID": "20", "Vencedor Escolhido": "EMPATE"},
        ])

        def claim(number):
            return claim_ticket_delivery(
                self.db_path, f"ticket-{number}", payload_hash
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            results = list(executor.map(claim, range(8)))

        self.assertEqual(1, sum(result[0] for result in results))

    def test_sent_composition_cannot_be_resent_with_another_name(self):
        payload_hash = delivery_fingerprint([
            {"ID": "10", "Vencedor Escolhido": "VISITANTE"},
        ])
        claimed, _, _ = claim_ticket_delivery(
            self.db_path, "ticket-original", payload_hash
        )
        self.assertTrue(claimed)
        mark_ticket_sent(self.db_path, "ticket-original", "1234")

        self.assertEqual(
            (False, "sent", "1234"),
            claim_ticket_delivery(self.db_path, "ticket-renomeado", payload_hash),
        )

    def test_ambiguous_timeout_is_not_retried_automatically(self):
        payload_hash = delivery_fingerprint([
            {"ID": "10", "Vencedor Escolhido": "MANDANTE"},
        ])
        claim_ticket_delivery(self.db_path, "ticket", payload_hash)
        mark_ticket_failed(self.db_path, "ticket", "read timeout", ambiguous=True)

        claimed, status, _ = claim_ticket_delivery(
            self.db_path, "ticket", payload_hash
        )
        self.assertFalse(claimed)
        self.assertEqual("unknown", status)


if __name__ == "__main__":
    unittest.main()

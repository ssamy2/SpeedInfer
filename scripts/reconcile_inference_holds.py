"""List or release crash-interrupted holds after the affected gateway has stopped.

Default: read-only inventory. Pass --release ID --gateway-stopped after verifying
that request is no longer executing. Never bulk refund active worker requests.
"""

import argparse
from datetime import UTC, datetime, timedelta

from sqlmodel import Session, select

from speedinfer.core.inference_billing import InferenceReservation, settle
from speedinfer.database.session import engine


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release")
    parser.add_argument("--gateway-stopped", action="store_true")
    args = parser.parse_args()
    with Session(engine) as session:
        if args.release:
            if not args.gateway_stopped:
                parser.error("Confirm the affected gateway is stopped with --gateway-stopped")
            hold = session.get(InferenceReservation, args.release)
            if not hold or hold.state != "reserved":
                parser.error("No pending hold with that ID")
            created = datetime.fromisoformat(hold.created_at)
            if created > datetime.now(UTC) - timedelta(minutes=15):
                parser.error("Hold is less than 15 minutes old; do not release an active request")
            settle(session, hold.id)
            print("Released", args.release)
        else:
            for hold in session.exec(
                select(InferenceReservation).where(InferenceReservation.state == "reserved")
            ):
                print(hold.id, hold.api_key_id, hold.amount, hold.created_at)


if __name__ == "__main__":
    main()

from fastapi import APIRouter, status

from ..deps import DbSession
from ..models import ContactSubmission
from ..schemas import ContactIn, MessageOut

router = APIRouter(tags=["contact"])


@router.post("/contact", response_model=MessageOut, status_code=status.HTTP_201_CREATED)
def submit_contact(body: ContactIn, db: DbSession) -> MessageOut:
    db.add(
        ContactSubmission(
            first_name=body.first_name.strip(),
            last_name=(body.last_name or "").strip() or None,
            email=body.email.lower(),
            phone=body.phone,
            message=body.message.strip(),
        )
    )
    db.commit()
    return MessageOut(message="Form submitted successfully")

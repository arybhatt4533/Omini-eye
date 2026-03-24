from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from . import models, schemas, database

router = APIRouter()

models.Base.metadata.create_all(bind=database.engine)

def get_db():
    db = database.SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.post("/persons", response_model=schemas.PersonOut)
def create_person(person: schemas.PersonCreate, db: Session = Depends(get_db)):
    db_person = models.Person(name=person.name, status=person.status)
    db.add(db_person)
    db.commit()
    db.refresh(db_person)
    return db_person

@router.get("/persons")
def list_persons(db: Session = Depends(get_db)):
    return db.query(models.Person).all()
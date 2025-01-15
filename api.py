from database import SessionLocal, LicensePlate
from fastapi import FastAPI, HTTPException, Depends
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
from fastapi.middleware.cors import CORSMiddleware 
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pathlib import Path

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_class=HTMLResponse)
async def read_root():
    html_file = Path("index.html").read_text()
    return html_file

# Pydantic models for request/response
class LicensePlateBase(BaseModel):
    plate_number: str
    owner_name: str
    cin: str
    notes: Optional[str] = None

class LicensePlateCreate(LicensePlateBase):
    pass

class LicensePlateResponse(LicensePlateBase):
    id: int
    created_at: datetime
    updated_at: datetime
    
    class Config:
        orm_mode = True

# Dependency to get database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# CRUD endpoints
@app.post("/plates/", response_model=LicensePlateResponse)
def create_plate(plate: LicensePlateCreate, db: Session = Depends(get_db)):
    db_plate = LicensePlate(
        plate_number=plate.plate_number.lower(),
        owner_name=plate.owner_name,
        cin=plate.cin,
        notes=plate.notes
    )
    db.add(db_plate)
    try:
        db.commit()
        db.refresh(db_plate)
    except:
        db.rollback()
        raise HTTPException(status_code=400, detail="Plate number or CIN already exists")
    return db_plate

@app.get("/plates/", response_model=List[LicensePlateResponse])
def read_plates(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    plates = db.query(LicensePlate).offset(skip).limit(limit).all()
    return plates

@app.get("/plates/{plate_id}", response_model=LicensePlateResponse)
def read_plate(plate_id: int, db: Session = Depends(get_db)):
    plate = db.query(LicensePlate).filter(LicensePlate.id == plate_id).first()
    if plate is None:
        raise HTTPException(status_code=404, detail="Plate not found")
    return plate

@app.put("/plates/{plate_id}", response_model=LicensePlateResponse)
def update_plate(plate_id: int, plate: LicensePlateCreate, db: Session = Depends(get_db)):
    db_plate = db.query(LicensePlate).filter(LicensePlate.id == plate_id).first()
    if db_plate is None:
        raise HTTPException(status_code=404, detail="Plate not found")
    
    db_plate.plate_number = plate.plate_number.lower()
    db_plate.owner_name = plate.owner_name
    db_plate.cin = plate.cin
    db_plate.notes = plate.notes
    db_plate.updated_at = datetime.utcnow()
    
    try:
        db.commit()
        db.refresh(db_plate)
    except:
        db.rollback()
        raise HTTPException(status_code=400, detail="Error updating plate")
    return db_plate

@app.delete("/plates/{plate_id}")
def delete_plate(plate_id: int, db: Session = Depends(get_db)):
    db_plate = db.query(LicensePlate).filter(LicensePlate.id == plate_id).first()
    if db_plate is None:
        raise HTTPException(status_code=404, detail="Plate not found")
    
    db.delete(db_plate)
    db.commit()
    return {"message": "Plate deleted successfully"}
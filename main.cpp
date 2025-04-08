#include <Eigen/Eigen>
#include <iostream>
#include <iomanip>
using namespace std;





Eigen::Vector2d PALU(Eigen::Matrix2d A, Eigen::Vector2d b)
{  Eigen::PartialPivLU<Eigen::MatrixXd> lu(A);
   Eigen::Matrix2d P = lu.permutationP();
   Eigen::Matrix2d L = Eigen::Matrix2d::Identity();
   L(1,0)= lu.matrixLU()(1,0);     
   Eigen::Vector2d Pb = P * b;
   Eigen::Vector2d y = L.triangularView<Eigen::Lower>().solve(Pb); //applico sost all'indietro 
   Eigen::Vector2d x = lu.matrixLU().triangularView<Eigen::Upper>().solve(y); // in avanti
   return x;
}
  
Eigen::Vector2d QR(Eigen::Matrix2d A, Eigen::Vector2d b)
{  Eigen::HouseholderQR<Eigen::MatrixXd> qr(A);
   Eigen::MatrixXd Q = qr.householderQ();
   Eigen::MatrixXd R = qr.matrixQR().triangularView<Eigen::Upper>();
   Eigen::MatrixXd Qtranspose = Q.transpose();
   Eigen::Vector2d y = Qtranspose * b;
   Eigen::Vector2d x = R.triangularView<Eigen::Upper>().solve(y);
   return x;
   
}
 


int main()

{   
    Eigen::Matrix2d A1;
    Eigen::Matrix2d A2;
    Eigen::Matrix2d A3;
    A1 << 5.547001962252291e-01, -3.770900990025203e-02,
          8.320502943378437e-01, -9.992887623566787e-01;
    A2 << 5.547001962252291e-01, -5.540607316466765e-01,
          8.320502943378437e-01, -8.324762492991313e-01;
    A3 << 5.547001962252291e-01, -5.547001955851905e-01,
          8.320502943378437e-01, -8.320502947645361e-01;
    Eigen::Vector2d  b1; 
    Eigen::Vector2d  b2;
    Eigen::Vector2d b3;
    b1 << -5.169911863249772e-01, 1.672384680188350e-01;
    b2 << -6.394645785530173e-04, 4.259549612877223e-04;
    b3 << -6.400391328043042e-10, 4.266924591433963e-10;
    
    Eigen::Vector2d x1_palu = PALU(A1,b1);
    Eigen::Vector2d x1_qr= QR(A1,b1);
    Eigen::Vector2d x2_palu= PALU(A2,b2);
    Eigen::Vector2d x2_qr = QR(A2,b2);
    Eigen::Vector2d x3_palu = PALU(A3,b3);
    Eigen::Vector2d x3_qr= QR(A3,b3);
    Eigen::Vector2d x_esatto;
    x_esatto << -1, -1;

    cout << "soluzione sistema 1 con PALU:\n" << std::scientific << x1_palu << endl;
    cout << "err relativo sistema 1 con PALU:\n" << std::scientific << ((x1_palu-x_esatto).norm()) / (x_esatto.norm()) << endl;
    cout << "e con QR:\n" << std::scientific <<  x1_qr << endl;
    cout << "err relativo sistema 1 con QR:\n" << std::scientific << ((x1_qr-x_esatto).norm()) / (x_esatto.norm()) << endl;
    cout << "soluzione sistema 2 con PALU:\n" << std::scientific << x2_palu << endl;
    cout << "err relativo sistema 2 con PALU:\n" << std::scientific << ((x2_palu-x_esatto).norm()) / (x_esatto.norm()) << endl;
    cout << "e con QR:\n" << std::scientific << x2_qr << endl;
    cout << "err relativo sistema 2 con QR:\n" << std::scientific << ((x2_qr-x_esatto).norm()) / (x_esatto.norm()) << endl;
    cout << "soluzione sistema 3 con PALU:\n" <<  std::scientific << x3_palu << endl;
    cout << "err relativo sistema 3 con PALU:\n" << std::scientific << ((x3_palu-x_esatto).norm()) / (x_esatto.norm()) << endl;
    cout << "e con QR:\n" << std::scientific << x3_qr << endl;
    cout << "err relativo sistema 3 con QR:\n" << std::scientific << ((x3_qr-x_esatto).norm()) / (x_esatto.norm()) << endl;

    return 0;
}

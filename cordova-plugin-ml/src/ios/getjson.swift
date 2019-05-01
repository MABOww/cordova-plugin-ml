import Foundation


//test data
"""let json = """

class GetData{
func GetJson(Dataset: String){
    //Json format definition
    struct UserInfo: Codable {
        struct Dataset: Codable {
            let id: Int
            let name: String
            
            struct DataArray: Codable {
                let thumbnailFilePath: String
                let imageFilePath: String
                let timeStamp: String
                let tag: String
            }
            let dataArray: [DataArray]
            
        }
        let dataSets: [Dataset]
    }
    // Json data get
    let decoder = JSONDecoder()
    do {
        let userInfo = try decoder.decode(UserInfo.self, from: json)
        print(userInfo)
    } catch DecodingError.keyNotFound(let key, let context) {
        print("keyNotFound: \(key): \(context)")
    } catch {
        print("\(error.localizedDescription)")
    }
    
}
}
